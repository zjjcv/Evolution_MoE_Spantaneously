"""
Collect intermediate layer outputs (proxies) from Gemma3-4B-Instruct on ARC dataset.

This script samples 10 questions from ARC-Easy and ARC-Challenge,
runs inference with Gemma3-4B-Instruct with auto-stop at EOS token (max 2048 tokens),
collects L2 norms and complete vectors across the generation timeline for:
- al: Attention outputs (per-head)
- ml: MLP outputs (per-layer)
- al + ml: Combined attention and MLP outputs (per-layer)

Output: 6 CSV files per difficulty (easy/challenge) with time series data

Usage:
    python src/3_proxy_collection_arc.py
"""

import copy
import json
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# ============== Configuration ==============
# 模型路径（可通过环境变量 MODEL_PATH 覆盖）
MODEL_PATH = os.environ.get("MODEL_PATH", "/root/data1/zjj/Neurlps2026/Checkpoints/Meta-Llama-3.1-8B-Instruct")

# 输出目录（可通过环境变量 OUTPUT_DIR 覆盖）
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/root/data1/zjj/Neurlps2026/Results/Llama-3.1-8B-Instruct/data")

ARC_DATA_DIR = "/root/data1/zjj/Neurlps2026/Dataset/ai2_arc"
NUM_SAMPLES = 10  # Number of questions to sample per difficulty
RANDOM_SEED = 42
MAX_NEW_TOKENS = 2048  # Maximum tokens to generate


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_arc_questions(json_path: str, num_samples: int, seed: int) -> List[Dict]:
    """Load and sample ARC questions from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    random.seed(seed)
    sampled = random.sample(data, min(num_samples, len(data)))
    return sampled


def create_arc_prompt(question: str, choices: str) -> str:
    """Create prompt template for ARC multiple choice questions."""
    prompt = f"""Question: {question}

Choices:
{choices}

Answer:"""
    return prompt


class ProxyCollector:
    """Collect intermediate layer outputs during model generation as time series."""

    def __init__(self, model, tokenizer, num_layers: int, num_heads: int, hidden_size: int, max_steps: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.max_steps = max_steps

        # Time series buffers: dict[layer/head] -> list of L2 norms
        self.al_buffer = {(i, j): [] for i in range(num_layers) for j in range(num_heads)}  # per-head
        self.ml_buffer = {i: [] for i in range(num_layers)}  # per-layer
        self.al_plus_ml_buffer = {i: [] for i in range(num_layers)}  # per-layer

        # Vector buffers: dict[layer/head] -> list of complete vectors (as numpy arrays)
        self.al_vector_buffer = {(i, j): [] for i in range(num_layers) for j in range(num_heads)}  # per-head
        self.ml_vector_buffer = {i: [] for i in range(num_layers)}  # per-layer
        self.al_plus_ml_vector_buffer = {i: [] for i in range(num_layers)}  # per-layer

        # Storage for current step al output (after o_proj)
        self._current_step_al_output = {i: None for i in range(num_layers)}
        self._current_step_ml_vector = {}

        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture intermediate outputs at each generation step."""
        self.hooks = []

        # Detect model structure
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'language_model'):
            # Gemma3: model.model.language_model.layers
            layers = self.model.model.language_model.layers
        elif hasattr(self.model, 'language_model'):
            layers = self.model.language_model.layers
        elif hasattr(self.model, 'model'):
            layers = self.model.model.layers
        else:
            raise ValueError(f"Unsupported model structure: {type(self.model)}")

        for layer_idx in range(self.num_layers):
            layer = layers[layer_idx]

            # Hook 1: Capture attention output (al) after self_attn
            def get_al_hook(layer_idx):
                def hook(module, input, output):
                    # output is a tuple, take first element (hidden_states)
                    attn_out = output[0] if isinstance(output, tuple) else output
                    # Reshape to get per-head outputs: [num_heads, head_dim]
                    head_dim = attn_out.shape[-1] // self.num_heads
                    reshaped = attn_out[0, -1, :].view(self.num_heads, head_dim)

                    # Compute L2 norm and store complete vector for each head
                    for head_idx in range(self.num_heads):
                        head_vector = reshaped[head_idx]  # [head_dim]
                        l2_norm = torch.linalg.norm(head_vector).item()
                        self.al_buffer[(layer_idx, head_idx)].append(l2_norm)
                        # Store complete vector as numpy array (convert to float32 first for bfloat16 compatibility)
                        self.al_vector_buffer[(layer_idx, head_idx)].append(head_vector.detach().cpu().float().numpy())
                return hook
            self.hooks.append(layer.self_attn.register_forward_hook(get_al_hook(layer_idx)))

            # Hook 2: Capture MLP output (ml)
            def get_ml_hook(layer_idx):
                def hook(module, input, output):
                    # output of mlp: [batch, seq, hidden_size]
                    mlp_out = output[0] if isinstance(output, tuple) else output
                    # Take last token
                    ml_vector = mlp_out[0, -1, :]  # [hidden_size]
                    l2_norm = torch.linalg.norm(ml_vector).item()
                    self.ml_buffer[layer_idx].append(l2_norm)
                    # Store complete vector as numpy array (convert to float32 first for bfloat16 compatibility)
                    self.ml_vector_buffer[layer_idx].append(ml_vector.detach().cpu().float().numpy())

                    # Store ml vector for al+ml computation
                    self._current_step_ml_vector[layer_idx] = ml_vector.detach().clone()
                return hook
            self.hooks.append(layer.mlp.register_forward_hook(get_ml_hook(layer_idx)))

            # Hook 3: Capture attention block output (after attention + residual) to get full al
            def get_attention_block_hook(layer_idx):
                def hook(module, input, output):
                    # output is tuple: (hidden_states, ...)
                    attn_output = output[0] if isinstance(output, tuple) else output
                    # Take last token: [hidden_size]
                    self._current_step_al_output[layer_idx] = attn_output[0, -1, :].detach().clone()

                    # Now compute al+ml if both are available
                    if self._current_step_al_output[layer_idx] is not None and layer_idx in self._current_step_ml_vector:
                        al_vector = self._current_step_al_output[layer_idx]
                        ml_vector = self._current_step_ml_vector[layer_idx]

                        # Vector sum: al + ml
                        combined_vector = al_vector + ml_vector
                        # Compute L2 norm of combined vector
                        combined_norm = torch.linalg.norm(combined_vector).item()
                        self.al_plus_ml_buffer[layer_idx].append(combined_norm)
                        # Store complete vector as numpy array (convert to float32 first for bfloat16 compatibility)
                        self.al_plus_ml_vector_buffer[layer_idx].append(combined_vector.detach().cpu().float().numpy())

                    # Clear ml vector for next step
                    if layer_idx in self._current_step_ml_vector:
                        del self._current_step_ml_vector[layer_idx]
                return hook
            self.hooks.append(layer.register_forward_hook(get_attention_block_hook(layer_idx)))

    def clear(self):
        """Clear time series buffers."""
        for key in self.al_buffer:
            self.al_buffer[key].clear()
        for key in self.ml_buffer:
            self.ml_buffer[key].clear()
        for key in self.al_plus_ml_buffer:
            self.al_plus_ml_buffer[key].clear()
        for key in self.al_vector_buffer:
            self.al_vector_buffer[key].clear()
        for key in self.ml_vector_buffer:
            self.ml_vector_buffer[key].clear()
        for key in self.al_plus_ml_vector_buffer:
            self.al_plus_ml_vector_buffer[key].clear()
        for key in self._current_step_al_output:
            self._current_step_al_output[key] = None
        self._current_step_ml_vector = {}

    def get_time_series_data(self):
        """Return collected time series data as dictionaries (deep copy)."""
        return {
            'al': copy.deepcopy(self.al_buffer),
            'ml': copy.deepcopy(self.ml_buffer),
            'al_plus_ml': copy.deepcopy(self.al_plus_ml_buffer),
            'al_vector': copy.deepcopy(self.al_vector_buffer),
            'ml_vector': copy.deepcopy(self.ml_vector_buffer),
            'al_plus_ml_vector': copy.deepcopy(self.al_plus_ml_vector_buffer)
        }

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()


def save_all_questions_time_series(all_ts: dict, output_path: str, proxy_type: str) -> None:
    """Save time series data for all questions.

    Args:
        all_ts: dict mapping question_id -> time_series_dict
        output_path: Path to save CSV
        proxy_type: Type of proxy ('al', 'ml', 'al_plus_ml')
    """
    records = []

    for q_idx in range(len(all_ts)):
        ts_dict = all_ts[q_idx]

        if proxy_type == 'al':
            # Per-head: (layer, head) -> list of L2 norms
            for (layer_idx, head_idx), time_series in sorted(ts_dict.items()):
                record = {
                    'question_id': q_idx,
                    'layer': layer_idx,
                    'head': head_idx
                }
                for step_idx, value in enumerate(time_series):
                    record[f'step_{step_idx+1}'] = value
                records.append(record)
        else:
            # Per-layer: layer -> list of L2 norms
            for layer_idx, time_series in sorted(ts_dict.items()):
                record = {
                    'question_id': q_idx,
                    'layer': layer_idx
                }
                for step_idx, value in enumerate(time_series):
                    record[f'step_{step_idx+1}'] = value
                records.append(record)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    num_steps = len(next(iter(all_ts[0].values()))) if all_ts else 0
    print(f"Saved {output_path}: {len(all_ts)} questions × {len(records) // len(all_ts)} components × {num_steps} steps")


def save_all_questions_vectors(all_vectors: dict, output_path: str, proxy_type: str) -> None:
    """Save complete vector data for all questions.

    Args:
        all_vectors: dict mapping question_id -> vector_dict
        output_path: Path to save CSV
        proxy_type: Type of proxy ('al', 'ml', 'al_plus_ml')
    """
    records = []

    for q_idx in range(len(all_vectors)):
        vector_dict = all_vectors[q_idx]

        if proxy_type == 'al':
            # Per-head: (layer, head) -> list of vectors
            for (layer_idx, head_idx), vector_list in sorted(vector_dict.items()):
                for step_idx, vector in enumerate(vector_list):
                    record = {
                        'question_id': q_idx,
                        'step': step_idx + 1,
                        'layer': layer_idx,
                        'head': head_idx
                    }
                    # Add each dimension as a separate column
                    for dim_idx, value in enumerate(vector):
                        record[f'dim_{dim_idx}'] = float(value)
                    records.append(record)
        else:
            # Per-layer: layer -> list of vectors
            for layer_idx, vector_list in sorted(vector_dict.items()):
                for step_idx, vector in enumerate(vector_list):
                    record = {
                        'question_id': q_idx,
                        'step': step_idx + 1,
                        'layer': layer_idx
                    }
                    # Add each dimension as a separate column
                    for dim_idx, value in enumerate(vector):
                        record[f'dim_{dim_idx}'] = float(value)
                    records.append(record)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    num_steps = len(next(iter(all_vectors[0].values()))) if all_vectors else 0
    num_components = len(all_vectors[0]) if all_vectors else 0  # Number of (layer, head) pairs or layers
    print(f"Saved {output_path}: {len(all_vectors)} questions × {num_components} components × (variable steps, avg ~{num_steps})")


def process_difficulty(model, tokenizer, difficulty: str, json_path: str, num_layers: int, num_heads: int, hidden_size: int):
    """Process one difficulty level (easy or challenge)."""
    print(f"\n{'=' * 60}")
    print(f"Processing ARC-{difficulty.capitalize()}")
    print(f"{'=' * 60}")

    # Load questions
    print(f"\n[1/3] Loading ARC-{difficulty.capitalize()} questions (sampling {NUM_SAMPLES})...")
    questions = load_arc_questions(json_path, NUM_SAMPLES, RANDOM_SEED)
    print(f"   Loaded: {len(questions)} questions")

    # Create output directory for this difficulty
    output_dir = os.path.join(OUTPUT_DIR, difficulty)
    os.makedirs(output_dir, exist_ok=True)

    # Create collector
    collector = ProxyCollector(model, tokenizer, num_layers, num_heads, hidden_size, MAX_NEW_TOKENS)

    # Collect data - one question at a time
    all_al_time_series = {}
    all_ml_time_series = {}
    all_al_plus_ml_time_series = {}
    effective_lengths = {}  # question_id -> effective_length

    print(f"\n[2/3] Collecting time series proxies from {len(questions)} questions...")
    print(f"   Auto-stopping at EOS (max {MAX_NEW_TOKENS} tokens per question)...")

    for q_idx, item in enumerate(tqdm(questions, desc=f"ARC-{difficulty.capitalize()}")):
        prompt = create_arc_prompt(item['question'], item['choices'])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs['input_ids'].shape[1]  # Track input length
        collector.clear()

        # Get stop token IDs based on model type
        eos_token_id = tokenizer.eos_token_id
        stop_token_ids = [eos_token_id]

        # Try to get additional stop tokens
        try:
            # For Qwen3: <|im_end|>
            if '<|im_end|>' in tokenizer.get_vocab():
                im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
                stop_token_ids.append(im_end_id)
        except:
            pass

        # Generate tokens (hooks will collect L2 norms at each step)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=stop_token_ids,  # Stop at any of the stop tokens
                use_cache=True,
                do_sample=False,
                return_dict_in_generate=True  # Get actual output
            )

        # Calculate actual generation length from output
        generated_ids = outputs.sequences[0]
        effective_length = len(generated_ids) - input_len
        effective_lengths[q_idx] = effective_length

        # Store time series data for this question
        ts_data = collector.get_time_series_data()
        all_al_time_series[q_idx] = ts_data['al'].copy()
        all_ml_time_series[q_idx] = ts_data['ml'].copy()
        all_al_plus_ml_time_series[q_idx] = ts_data['al_plus_ml'].copy()

    collector.remove_hooks()

    # Save data
    print(f"\n[3/3] Saving data to {output_dir}...")

    # Save L2 norm files
    al_path = os.path.join(output_dir, f"arc_{difficulty}_al.csv")
    save_all_questions_time_series(all_al_time_series, al_path, 'al')

    ml_path = os.path.join(output_dir, f"arc_{difficulty}_ml.csv")
    save_all_questions_time_series(all_ml_time_series, ml_path, 'ml')

    al_plus_ml_path = os.path.join(output_dir, f"arc_{difficulty}_al_plus_ml.csv")
    save_all_questions_time_series(all_al_plus_ml_time_series, al_plus_ml_path, 'al_plus_ml')

    # Save effective lengths
    effective_lengths_path = os.path.join(output_dir, f"arc_{difficulty}_effective_lengths.csv")
    effective_lengths_df = pd.DataFrame([
        {'question_id': q_id, 'effective_length': length}
        for q_id, length in effective_lengths.items()
    ])
    effective_lengths_df.to_csv(effective_lengths_path, index=False)
    print(f"Saved {effective_lengths_path}: {len(effective_lengths)} questions")
    print(f"  Average effective length: {np.mean(list(effective_lengths.values())):.1f} steps")
    print(f"  Min/Max: {min(effective_lengths.values())} / {max(effective_lengths.values())} steps")


def main():
    """Main execution function."""
    set_seed(RANDOM_SEED)

    print("=" * 60)
    print(f"ARC Proxy Collection with {os.path.basename(MODEL_PATH)}")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    print(f"\n[0/3] Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.eval()

    # Get model config
    config = model.config
    if hasattr(config, 'text_config'):
        text_config = config.text_config
        num_layers = text_config.num_hidden_layers
        hidden_size = text_config.hidden_size
    else:
        num_layers = config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else len(model.model.layers)
        hidden_size = config.hidden_size if hasattr(config, 'hidden_size') else 4096

    # Detect model structure (different for Gemma3 vs Qwen3)
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        # Gemma3: model.model.language_model.layers
        layers = model.model.language_model.layers
        first_layer_attn = layers[0].self_attn
    elif hasattr(model, 'language_model'):
        layers = model.language_model.layers
        first_layer_attn = layers[0].self_attn
    elif hasattr(model, 'model'):
        layers = model.model.layers
        first_layer_attn = layers[0].self_attn
    else:
        raise ValueError(f"Unsupported model structure: {type(model)}")

    # Get number of attention heads
    head_dim = first_layer_attn.head_dim if hasattr(first_layer_attn, 'head_dim') else hidden_size // config.num_attention_heads
    q_proj_out = first_layer_attn.q_proj.out_features
    num_heads = q_proj_out // head_dim

    print(f"   Model: {num_layers} layers, {num_heads} heads/layer, hidden_size={hidden_size}")

    # Process both difficulties
    difficulties = [
        ('easy', os.path.join(ARC_DATA_DIR, 'ARC-Easy_json', 'test-00000-of-00001.json')),
        ('challenge', os.path.join(ARC_DATA_DIR, 'ARC-Challenge_json', 'test-00000-of-00001.json'))
    ]

    for difficulty, json_path in difficulties:
        if not os.path.exists(json_path):
            print(f"\nWarning: {json_path} not found, skipping {difficulty}")
            continue
        process_difficulty(model, tokenizer, difficulty, json_path, num_layers, num_heads, hidden_size)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files for each difficulty (easy/challenge):")
    print("L2 norm files:")
    print("  - arc_{difficulty}_al.csv")
    print("  - arc_{difficulty}_ml.csv")
    print("  - arc_{difficulty}_al_plus_ml.csv")
    print("\nOther:")
    print("  - arc_{difficulty}_effective_lengths.csv")


if __name__ == "__main__":
    main()
