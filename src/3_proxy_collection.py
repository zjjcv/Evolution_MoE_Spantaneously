"""
Collect intermediate layer outputs (proxies) from Gemma-3-4B-Instruct on GSM8K dataset.

Samples 10 questions from GSM8K test set, runs inference with auto-stop at EOS
(max 2048 tokens), collects L2 norms across the generation timeline for:
- al: Attention outputs (per-head)
- ml: MLP outputs (per-layer)
- al + ml: Combined attention and MLP outputs (per-layer)

Output: 4 CSV files with time series data (no vectors)

Usage:
    python src/3_proxy_collection.py
"""

import json
import os
import random
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# ============== Configuration ==============
MODEL_PATH = "/root/data1/zjj/Neurlps2026/Checkpoints/Meta-Llama-3.1-8B-Instruct"
GSM8K_DATA_DIR = "/root/data1/zjj/Neurlps2026/Dataset/gsm8k"
OUTPUT_DIR = "/root/data1/zjj/Neurlps2026/Results/Llama-3.1-8B-Instruct/data/L2_Norm"

NUM_SAMPLES = 10
RANDOM_SEED = 42
MAX_NEW_TOKENS = 2048


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_gsm8k_questions(data_dir: str, num_samples: int, seed: int) -> List[Dict]:
    """Load and sample questions from GSM8K test set."""
    random.seed(seed)
    test_file = os.path.join(data_dir, "json", "test.json")

    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return random.sample(data, min(num_samples, len(data)))


def create_gsm8k_prompt(question: str) -> str:
    """Create prompt template for GSM8K question."""
    return f"Question: {question}\nAnswer:"


class ProxyCollector:
    """Collect intermediate layer outputs during model generation as time series."""

    def __init__(self, model, tokenizer, num_layers: int, num_heads: int, hidden_size: int, max_steps: int = 2048):
        self.model = model
        self.tokenizer = tokenizer
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.max_steps = max_steps

        # Time series buffers: dict[layer/head] -> list of L2 norms
        self.al_buffer = {(i, j): [] for i in range(num_layers) for j in range(num_heads)}
        self.ml_buffer = {i: [] for i in range(num_layers)}
        self.al_plus_ml_buffer = {i: [] for i in range(num_layers)}

        # Storage for current step outputs
        self._current_step_al_output = {i: None for i in range(num_layers)}
        self._current_step_ml_vector = {}

        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture intermediate outputs at each generation step."""
        self.hooks = []

        if hasattr(self.model, 'language_model'):
            layers = self.model.language_model.layers
        elif hasattr(self.model, 'model'):
            layers = self.model.model.layers
        else:
            raise ValueError(f"Unsupported model structure: {type(self.model)}")

        for layer_idx in range(self.num_layers):
            layer = layers[layer_idx]

            # Hook 1: Capture attention output (al) per head
            def get_al_hook(layer_idx):
                def hook(module, input, output):
                    attn_out = output[0] if isinstance(output, tuple) else output
                    head_dim = attn_out.shape[-1] // self.num_heads
                    reshaped = attn_out[0, -1, :].view(self.num_heads, head_dim)
                    for head_idx in range(self.num_heads):
                        head_vector = reshaped[head_idx]
                        self.al_buffer[(layer_idx, head_idx)].append(
                            torch.linalg.norm(head_vector).item()
                        )
                return hook
            self.hooks.append(layer.self_attn.register_forward_hook(get_al_hook(layer_idx)))

            # Hook 2: Capture MLP output (ml)
            def get_ml_hook(layer_idx):
                def hook(module, input, output):
                    mlp_out = output[0] if isinstance(output, tuple) else output
                    ml_vector = mlp_out[0, -1, :]
                    self.ml_buffer[layer_idx].append(
                        torch.linalg.norm(ml_vector).item()
                    )
                    self._current_step_ml_vector[layer_idx] = ml_vector.detach().clone()
                return hook
            self.hooks.append(layer.mlp.register_forward_hook(get_ml_hook(layer_idx)))

            # Hook 3: Capture attention block output and compute al+ml
            def get_attention_block_hook(layer_idx):
                def hook(module, input, output):
                    attn_output = output[0] if isinstance(output, tuple) else output
                    self._current_step_al_output[layer_idx] = attn_output[0, -1, :].detach().clone()

                    if self._current_step_al_output[layer_idx] is not None and layer_idx in self._current_step_ml_vector:
                        al_vector = self._current_step_al_output[layer_idx]
                        ml_vector = self._current_step_ml_vector[layer_idx]
                        combined_vector = al_vector + ml_vector
                        self.al_plus_ml_buffer[layer_idx].append(
                            torch.linalg.norm(combined_vector).item()
                        )
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
        for key in self._current_step_al_output:
            self._current_step_al_output[key] = None
        self._current_step_ml_vector = {}

    def get_time_series_data(self):
        """Return collected time series data as dictionaries."""
        return {
            'al': dict(self.al_buffer),
            'ml': dict(self.ml_buffer),
            'al_plus_ml': dict(self.al_plus_ml_buffer),
        }

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()


def save_time_series(all_ts: dict, output_path: str, proxy_type: str):
    """Save time series data for all questions to CSV."""
    records = []

    for q_idx in range(len(all_ts)):
        ts_dict = all_ts[q_idx]
        if proxy_type == 'al':
            for (layer_idx, head_idx), time_series in sorted(ts_dict.items()):
                record = {'question_id': q_idx, 'layer': layer_idx, 'head': head_idx}
                for step_idx, value in enumerate(time_series):
                    record[f'step_{step_idx+1}'] = value
                records.append(record)
        else:
            for layer_idx, time_series in sorted(ts_dict.items()):
                record = {'question_id': q_idx, 'layer': layer_idx}
                for step_idx, value in enumerate(time_series):
                    record[f'step_{step_idx+1}'] = value
                records.append(record)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    num_steps = len(next(iter(all_ts[0].values()))) if all_ts else 0
    print(f"Saved {output_path}: {len(all_ts)} questions x {len(records) // max(len(all_ts),1)} components x {num_steps} steps")


def main():
    set_seed(RANDOM_SEED)

    print("=" * 60)
    print(f"GSM8K Proxy Collection with {os.path.basename(MODEL_PATH)}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    print(f"\n[1/4] Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
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
        num_layers = config.num_hidden_layers
        hidden_size = config.hidden_size

    if hasattr(model, 'language_model'):
        layers = model.language_model.layers
    elif hasattr(model, 'model'):
        layers = model.model.layers
    else:
        raise ValueError(f"Unsupported model structure: {type(model)}")

    first_layer_attn = layers[0].self_attn
    head_dim = first_layer_attn.head_dim if hasattr(first_layer_attn, 'head_dim') else hidden_size // config.num_attention_heads
    q_proj_out = first_layer_attn.q_proj.out_features
    num_heads = q_proj_out // head_dim

    print(f"   Model: {num_layers} layers, {num_heads} heads/layer, hidden_size={hidden_size}")

    # Load GSM8K questions
    print(f"\n[2/4] Loading {NUM_SAMPLES} GSM8K questions...")
    questions = load_gsm8k_questions(GSM8K_DATA_DIR, NUM_SAMPLES, RANDOM_SEED)
    print(f"   Loaded: {len(questions)} questions")

    # Process questions
    print(f"\n[3/4] Collecting time series proxies from {len(questions)} questions...")
    collector = ProxyCollector(model, tokenizer, num_layers, num_heads, hidden_size, MAX_NEW_TOKENS)

    all_al_time_series = {}
    all_ml_time_series = {}
    all_al_plus_ml_time_series = {}
    effective_lengths = {}

    for q_idx, question in enumerate(tqdm(questions, desc="Collecting proxies")):
        prompt = create_gsm8k_prompt(question['question'])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs['input_ids'].shape[1]
        collector.clear()

        eos_token_id = tokenizer.eos_token_id
        stop_token_ids = [eos_token_id]
        try:
            if '<end_of_turn>' in tokenizer.get_vocab():
                end_of_turn_id = tokenizer.convert_tokens_to_ids('<end_of_turn>')
                stop_token_ids.append(end_of_turn_id)
        except:
            pass

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=stop_token_ids,
                use_cache=True,
                do_sample=False,
                return_dict_in_generate=True
            )

        effective_length = len(outputs.sequences[0]) - input_len
        effective_lengths[q_idx] = effective_length

        ts_data = collector.get_time_series_data()
        all_al_time_series[q_idx] = ts_data['al']
        all_ml_time_series[q_idx] = ts_data['ml']
        all_al_plus_ml_time_series[q_idx] = ts_data['al_plus_ml']

    collector.remove_hooks()

    # Save time series
    print(f"\n[4/4] Saving time series...")

    save_time_series(all_al_time_series, os.path.join(OUTPUT_DIR, "gsm8k_al.csv"), 'al')
    save_time_series(all_ml_time_series, os.path.join(OUTPUT_DIR, "gsm8k_ml.csv"), 'ml')
    save_time_series(all_al_plus_ml_time_series, os.path.join(OUTPUT_DIR, "gsm8k_al_plus_ml.csv"), 'al_plus_ml')

    # Save effective lengths
    effective_lengths_path = os.path.join(OUTPUT_DIR, "gsm8k_effective_lengths.csv")
    effective_lengths_df = pd.DataFrame([
        {'question_id': q_id, 'effective_length': length}
        for q_id, length in effective_lengths.items()
    ])
    effective_lengths_df.to_csv(effective_lengths_path, index=False)
    print(f"Saved {effective_lengths_path}: {len(effective_lengths)} questions")
    print(f"  Average effective length: {np.mean(list(effective_lengths.values())):.1f} steps")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - gsm8k_al.csv")
    print("  - gsm8k_ml.csv")
    print("  - gsm8k_al_plus_ml.csv")
    print("  - gsm8k_effective_lengths.csv")


if __name__ == "__main__":
    main()
