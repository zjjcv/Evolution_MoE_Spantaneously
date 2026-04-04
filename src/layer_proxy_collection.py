"""
Collect layer-wise proxy vectors from Gemma3-4B-Instruct on GSM8K dataset.

This script samples 10 questions from GSM8K, runs inference with Gemma3-4B-Instruct
with auto-stop at EOS token (max 2048 tokens), and collects averaged vectors
across all generation steps for:
- al: Attention outputs (per-head, averaged across steps)
- ml: MLP outputs (per-layer, averaged across steps)
- al + ml: Combined outputs (per-layer, averaged across steps)

Output: 3 CSV files with averaged vectors

Usage:
    python src/layer_proxy_collection.py
"""

import copy
import json
import os
import random
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# ============== Configuration ==============
MODEL_PATH = os.environ.get("MODEL_PATH", "/root/data1/zjj/Neurlps2026/Checkpoints/Meta-Llama-3.1-8B-Instruct")
GSM8K_DATA_DIR = "/root/data1/zjj/Neurlps2026/Dataset/gsm8k"
OUTPUT_DIR = "/root/data1/zjj/Neurlps2026/Results/Llama-3.1-8B-Instruct/data/residual_stream"

NUM_QUESTIONS = 10
RANDOM_SEED = 42
MAX_NEW_TOKENS = 2048


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_gsm8k_questions(data_dir: str, num_questions: int, seed: int) -> List[Dict]:
    """Load and sample GSM8K questions.

    Args:
        data_dir: Directory containing GSM8K data
        num_questions: Number of questions to sample
        seed: Random seed

    Returns:
        List of question dictionaries
    """
    random.seed(seed)

    # GSM8K json directory
    test_file = os.path.join(data_dir, "json", "test.json")

    if not os.path.exists(test_file):
        raise FileNotFoundError(f"GSM8K data not found at {test_file}")

    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Sample questions
    num_to_sample = min(num_questions, len(data))
    sampled = random.sample(data, num_to_sample)

    print(f"Sampled {num_to_sample} questions from GSM8K")
    return sampled


def create_gsm8k_prompt(question: str) -> str:
    """Create prompt for GSM8K question.

    Args:
        question: Question text (may contain question and answer)

    Returns:
        Formatted prompt
    """
    # GSM8K format: "Question: ... \nAnswer: ..."
    # Extract just the question part if needed
    if "Question:" in question:
        prompt = question
    else:
        prompt = f"Question: {question}\nAnswer:"

    return prompt


class LayerProxyCollector:
    """Collect layer-wise averaged vectors during model generation."""

    def __init__(self, model, tokenizer, num_layers: int, num_heads: int, hidden_size: int):
        self.model = model
        self.tokenizer = tokenizer
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        # Accumulator buffers for averaging
        # al: dict[(layer, head)] -> accumulated vector
        self.al_accumulator = {(i, j): np.zeros(hidden_size, dtype=np.float32)
                               for i in range(num_layers) for j in range(num_heads)}
        # ml: dict[layer] -> accumulated vector
        self.ml_accumulator = {i: np.zeros(hidden_size, dtype=np.float32)
                               for i in range(num_layers)}
        # al+ml: dict[layer] -> accumulated vector
        self.al_plus_ml_accumulator = {i: np.zeros(hidden_size, dtype=np.float32)
                                       for i in range(num_layers)}

        # Counters for averaging
        self.al_count = {(i, j): 0 for i in range(num_layers) for j in range(num_heads)}
        self.ml_count = {i: 0 for i in range(num_layers)}
        self.al_plus_ml_count = {i: 0 for i in range(num_layers)}

        # Hooks
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture intermediate outputs."""

        # Get model layers (handle both Gemma3 and Qwen3 architectures)
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'language_model'):
            # Gemma3 structure: model.model.language_model.layers
            layers = self.model.model.language_model.layers
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Qwen3 structure: model.model.layers
            layers = self.model.model.layers
        else:
            raise ValueError(f"Unsupported model structure: {type(self.model)}")

        # Register hooks for each layer
        for idx, layer in enumerate(layers):
            # Hook for MLP output
            mlp_hook = layer.mlp.register_forward_hook(
                lambda module, input, output, layer_idx=idx: self._ml_hook_fn(layer_idx, module, input, output)
            )
            self.hooks.append(mlp_hook)

            # Hook for attention output (will be captured separately)
            attn_hook = layer.self_attn.register_forward_hook(
                lambda module, input, output, layer_idx=idx: self._attn_hook_fn(layer_idx, module, input, output)
            )
            self.hooks.append(attn_hook)

    def _ml_hook_fn(self, layer_idx, module, input, output):
        """MLP hook function."""
        if isinstance(output, tuple):
            output = output[0]

        # Average over batch and sequence (take mean)
        output_np = output.detach().cpu().float().numpy()  # [batch, seq, hidden]
        output_mean = output_np.mean(axis=(0, 1))  # [hidden]

        self.ml_accumulator[layer_idx] += output_mean
        self.ml_count[layer_idx] += 1

    def _attn_hook_fn(self, layer_idx, module, input, output):
        """Attention hook function - captures attention output."""
        if isinstance(output, tuple):
            output = output[0]

        # The attention output is after projection: [batch, seq, hidden]
        output_np = output.detach().cpu().float().numpy()
        output_mean = output_np.mean(axis=(0, 1))  # [hidden]

        # For AL, we accumulate this (aggregated across heads)
        # Note: This gives us the sum of all heads' outputs projected to hidden_dim
        self.al_plus_ml_accumulator[layer_idx] += output_mean
        # We'll use ml_accumulator to get the ML contribution, then compute AL
        # AL = (AL+ML) - ML is not correct...

        # Actually, let's think again:
        # The attention output we get here is the final projected output
        # which is sum of all heads' projections
        # Let's call this "attn_output"

        # For the al_plus_ml_accumulator, we want: attn_output + mlp_output
        # So we need to store attn_output separately
        if not hasattr(self, '_attn_output_accumulator'):
            self._attn_output_accumulator = {i: np.zeros(self.hidden_size, dtype=np.float32)
                                             for i in range(self.num_layers)}
            self._attn_output_count = {i: 0 for i in range(self.num_layers)}

        self._attn_output_accumulator[layer_idx] += output_mean
        self._attn_output_count[layer_idx] += 1

    def collect(self, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS):
        """Run generation and collect layer outputs.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
        """
        # Reset accumulators
        self.al_accumulator = {(i, j): np.zeros(self.hidden_size, dtype=np.float32)
                               for i in range(self.num_layers) for j in range(self.num_heads)}
        self.ml_accumulator = {i: np.zeros(self.hidden_size, dtype=np.float32)
                               for i in range(self.num_layers)}
        self.al_plus_ml_accumulator = {i: np.zeros(self.hidden_size, dtype=np.float32)
                                       for i in range(self.num_layers)}
        self._attn_output_accumulator = {i: np.zeros(self.hidden_size, dtype=np.float32)
                                         for i in range(self.num_layers)}

        self.al_count = {(i, j): 0 for i in range(self.num_layers) for j in range(self.num_heads)}
        self.ml_count = {i: 0 for i in range(self.num_layers)}
        self.al_plus_ml_count = {i: 0 for i in range(self.num_layers)}
        self._attn_output_count = {i: 0 for i in range(self.num_layers)}

        # Encode prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate with sampling
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_hidden_states=False,
                return_dict_in_generate=False,
            )

        # Average the accumulated values
        num_steps = self.ml_count[0] if self.ml_count[0] > 0 else 1

        # Compute averaged vectors
        al_avg = {}
        ml_avg = {}
        al_plus_ml_avg = {}

        for i in range(self.num_layers):
            if self.ml_count[i] > 0:
                ml_avg[i] = self.ml_accumulator[i] / self.ml_count[i]
            else:
                ml_avg[i] = np.zeros(self.hidden_size, dtype=np.float32)

            if self._attn_output_count[i] > 0:
                attn_avg = self._attn_output_accumulator[i] / self._attn_output_count[i]
            else:
                attn_avg = np.zeros(self.hidden_size, dtype=np.float32)

            al_avg[i] = attn_avg  # Attention output (sum of heads)
            al_plus_ml_avg[i] = attn_avg + ml_avg[i]

        return al_avg, ml_avg, al_plus_ml_avg

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def save_results(all_al_vectors, all_ml_vectors, all_al_plus_ml_vectors,
                 num_layers, hidden_size, output_dir):
    """Save collected vectors to CSV files.

    Args:
        all_al_vectors: Dict[question_id][layer] -> vector
        all_ml_vectors: Dict[question_id][layer] -> vector
        all_al_plus_ml_vectors: Dict[question_id][layer] -> vector
        num_layers: Number of layers
        hidden_size: Hidden size
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    num_questions = len(all_al_vectors)

    # Prepare data for AL
    al_data = []
    for q_id in range(num_questions):
        for layer in range(num_layers):
            vector = all_al_vectors[q_id][layer]
            row = {'question_id': q_id, 'layer': layer}
            for dim in range(hidden_size):
                row[f'dim_{dim}'] = vector[dim]
            al_data.append(row)

    # Prepare data for ML
    ml_data = []
    for q_id in range(num_questions):
        for layer in range(num_layers):
            vector = all_ml_vectors[q_id][layer]
            row = {'question_id': q_id, 'layer': layer}
            for dim in range(hidden_size):
                row[f'dim_{dim}'] = vector[dim]
            ml_data.append(row)

    # Prepare data for AL+ML
    al_plus_ml_data = []
    for q_id in range(num_questions):
        for layer in range(num_layers):
            vector = all_al_plus_ml_vectors[q_id][layer]
            row = {'question_id': q_id, 'layer': layer}
            for dim in range(hidden_size):
                row[f'dim_{dim}'] = vector[dim]
            al_plus_ml_data.append(row)

    # Save to CSV
    al_df = pd.DataFrame(al_data)
    ml_df = pd.DataFrame(ml_data)
    al_plus_ml_df = pd.DataFrame(al_plus_ml_data)

    al_output = os.path.join(output_dir, "al.csv")
    ml_output = os.path.join(output_dir, "ml.csv")
    al_plus_ml_output = os.path.join(output_dir, "al_plus_ml.csv")

    al_df.to_csv(al_output, index=False)
    ml_df.to_csv(ml_output, index=False)
    al_plus_ml_df.to_csv(al_plus_ml_output, index=False)

    print(f"\nSaved results to:")
    print(f"  {al_output}")
    print(f"  {ml_output}")
    print(f"  {al_plus_ml_output}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Layer Proxy Collection for GSM8K")
    print("=" * 60)

    # Set seed
    set_seed(RANDOM_SEED)

    # Load model and tokenizer
    print(f"\nLoading model from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Use bfloat16 for Gemma3, float16 for others
    if "Gemma" in MODEL_PATH or "gemma" in MODEL_PATH:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True
    )

    # Model parameters (handle Gemma3's nested text_config)
    if hasattr(model.config, 'text_config'):
        config = model.config.text_config
    else:
        config = model.config

    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    hidden_size = config.hidden_size

    print(f"  Layers: {num_layers}")
    print(f"  Heads: {num_heads}")
    print(f"  Hidden size: {hidden_size}")

    # Load questions
    print(f"\nLoading GSM8K questions...")
    questions = load_gsm8k_questions(GSM8K_DATA_DIR, NUM_QUESTIONS, RANDOM_SEED)

    # Create collector
    collector = LayerProxyCollector(model, tokenizer, num_layers, num_heads, hidden_size)

    # Process each question
    all_al_vectors = {}
    all_ml_vectors = {}
    all_al_plus_ml_vectors = {}

    print(f"\nProcessing {len(questions)} questions...")
    for q_idx, question in enumerate(tqdm(questions)):
        # Extract question text
        if isinstance(question, dict):
            question_text = question.get('question', '')
        else:
            question_text = str(question)

        # Create prompt
        prompt = create_gsm8k_prompt(question_text)

        # Collect vectors
        al_avg, ml_avg, al_plus_ml_avg = collector.collect(prompt, MAX_NEW_TOKENS)

        all_al_vectors[q_idx] = al_avg
        all_ml_vectors[q_idx] = ml_avg
        all_al_plus_ml_vectors[q_idx] = al_plus_ml_avg

    # Remove hooks
    collector.remove_hooks()

    # Save results
    save_results(all_al_vectors, all_ml_vectors, all_al_plus_ml_vectors,
                num_layers, hidden_size, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("All Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
