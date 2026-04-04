"""
Collect layer-wise relative change data when ablating each layer (Multi-GPU).

This script samples 8 questions from GSM8K, runs inference on 8 GPUs in parallel,
and collects the relative impact of ablating each layer on subsequent layers.

Formula for l > s (layers after ablated layer s):
    ||(h_l+1 - h_l) - (¯h_l+1 - ¯h_l)||_2 / ||h_l+1 - h_l||_2

Where:
    h_l: normal residual stream at layer l
    ¯h_l: residual stream when layer s is ablated

Output: CSV file with columns: question_id, ablated_layer_s, affected_layer_l, relative_change

Usage:
    python src/layer_relative_change.py
"""

import json
import os
import random
from typing import Dict, List, Tuple
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============== Configuration ==============
MODEL_PATH = os.environ.get("MODEL_PATH", "/root/data1/zjj/Neurlps2026/Checkpoints/Meta-Llama-3.1-8B-Instruct")
GSM8K_DATA_DIR = "/root/data1/zjj/Neurlps2026/Dataset/gsm8k"
OUTPUT_DIR = "/root/data1/zjj/Neurlps2026/Results/Llama-3.1-8B-Instruct/data/IG_Relative/layer_relative_change"

NUM_QUESTIONS = 2
NUM_GPUS = 2
RANDOM_SEED = 42
MAX_NEW_TOKENS = 2048


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_gsm8k_questions(data_dir: str, num_questions: int, seed: int) -> List[Dict]:
    """Load and sample GSM8K questions."""
    random.seed(seed)

    test_file = os.path.join(data_dir, "json", "test.json")

    if not os.path.exists(test_file):
        raise FileNotFoundError(f"GSM8K data not found at {test_file}")

    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    num_to_sample = min(num_questions, len(data))
    sampled = random.sample(data, num_to_sample)

    return sampled


def create_gsm8k_prompt(question: str) -> str:
    """Create prompt for GSM8K question."""
    if "Question:" in question:
        return question
    else:
        return f"Question: {question}\nAnswer:"


def get_model_layers(model):
    """Detect and return the model's transformer layers."""
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        return model.model.language_model.layers
    elif hasattr(model, 'language_model'):
        return model.language_model.layers
    else:
        return model.model.layers


def process_single_question(gpu_id: int, q_idx: int, question_text: str, model_path: str,
                            max_new_tokens: int, output_dir: str) -> List[Dict]:
    """Process a single question on a specific GPU.

    Args:
        gpu_id: GPU device ID
        q_idx: Question index
        question_text: Question text
        model_path: Path to model weights
        max_new_tokens: Maximum new tokens to generate
        output_dir: Output directory (for temp files if needed)

    Returns:
        List of result dictionaries
    """
    # Set device for this process
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)

    print(f"[GPU {gpu_id}] Processing question {q_idx}...")

    # Load model on this GPU (bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True
    )

    if hasattr(model.config, 'text_config'):
        num_layers = model.config.text_config.num_hidden_layers
        hidden_size = model.config.text_config.hidden_size
    else:
        num_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
    model_layers = get_model_layers(model)

    # Storage for all results for this question
    all_results = []

    # Create prompt
    prompt = create_gsm8k_prompt(question_text)

    # Step 1: Collect normal deltas (baseline)
    print(f"[GPU {gpu_id}] Q{q_idx}: Collecting normal deltas...")

    layer_outputs = {}

    def create_normal_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            out_np = out.detach().cpu().float().numpy()
            out_mean = out_np.mean(axis=(0, 1))
            layer_outputs[layer_idx] = out_mean
        return hook

    hooks = []
    for idx, layer in enumerate(model_layers):
        hook = layer.register_forward_hook(create_normal_hook(idx))
        hooks.append(hook)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_hidden_states=False,
            return_dict_in_generate=False,
        )

    for hook in hooks:
        hook.remove()

    # Compute normal deltas
    normal_deltas = {}
    for layer in range(num_layers - 1):
        if layer in layer_outputs and (layer + 1) in layer_outputs:
            normal_deltas[layer] = layer_outputs[layer + 1] - layer_outputs[layer]

    # Step 2: For each ablated layer s, collect ablated deltas
    for ablated_layer in range(num_layers):
        if ablated_layer % 5 == 0:
            print(f"[GPU {gpu_id}] Q{q_idx}: Ablating layer {ablated_layer}/{num_layers}...")

        ablated_layer_outputs = {}

        def create_ablated_hook(layer_idx, ablated_s):
            def hook(module, input, output):
                if layer_idx == ablated_s:
                    # Ablate this layer: return input unchanged (skip layer)
                    if isinstance(input, tuple):
                        batch_size, seq_len, hidden = input[0].shape
                        return input[0]
                    else:
                        return input
                else:
                    if isinstance(output, tuple):
                        out = output[0]
                    else:
                        out = output
                    out_np = out.detach().cpu().float().numpy()
                    out_mean = out_np.mean(axis=(0, 1))
                    ablated_layer_outputs[layer_idx] = out_mean
                    return output
            return hook

        hooks = []
        for idx, layer in enumerate(model_layers):
            hook = layer.register_forward_hook(create_ablated_hook(idx, ablated_layer))
            hooks.append(hook)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_hidden_states=False,
                return_dict_in_generate=False,
            )

        for hook in hooks:
            hook.remove()

        # Compute relative changes for affected layers (l > ablated_layer)
        for affected_layer in range(ablated_layer + 1, num_layers):
            if affected_layer in ablated_layer_outputs and (affected_layer + 1) in ablated_layer_outputs:
                ablated_delta = ablated_layer_outputs[affected_layer + 1] - ablated_layer_outputs[affected_layer]
            elif affected_layer in ablated_layer_outputs:
                ablated_delta = ablated_layer_outputs[affected_layer]
            else:
                continue

            normal_delta = normal_deltas.get(affected_layer)
            if normal_delta is None:
                continue

            # Compute relative change
            diff = normal_delta - ablated_delta
            normal_norm = np.linalg.norm(normal_delta)

            if normal_norm < 1e-10:
                rel_change = 0.0
            else:
                rel_change = np.linalg.norm(diff) / normal_norm

            all_results.append({
                'question_id': q_idx,
                'ablated_layer_s': ablated_layer,
                'affected_layer_l': affected_layer,
                'relative_change': rel_change
            })

    # Cleanup
    del model
    torch.cuda.empty_cache()

    print(f"[GPU {gpu_id}] Q{q_idx}: Complete, collected {len(all_results)} records")
    return all_results


def save_results(all_results: List[Dict], output_dir: str):
    """Save collected results to CSV file."""
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(all_results)
    output_path = os.path.join(output_dir, "layer_relative_change.csv")

    df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"Saved results to: {output_path}")
    print(f"  Total records: {len(df):,}")
    print(f"  Questions: {df['question_id'].nunique()}")
    print(f"  Ablated layers: {sorted(df['ablated_layer_s'].unique())}")
    print(f"  Relative change range: {df['relative_change'].min():.6f} - {df['relative_change'].max():.6f}")
    print(f"  Mean relative change: {df['relative_change'].mean():.6f}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Layer Relative Change Collection (Multi-GPU)")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  GPUs: {NUM_GPUS}")
    print(f"  Questions: {NUM_QUESTIONS}")
    print(f"  Model: {MODEL_PATH}")

    # Set seed
    set_seed(RANDOM_SEED)

    # Load questions
    print(f"\nLoading GSM8K questions...")
    questions = load_gsm8k_questions(GSM8K_DATA_DIR, NUM_QUESTIONS, RANDOM_SEED)
    print(f"  Sampled {len(questions)} questions")

    # Prepare question data for each GPU
    question_data = []
    for i, q in enumerate(questions):
        if isinstance(q, dict):
            q_text = q.get('question', '')
        else:
            q_text = str(q)
        question_data.append((i, q_text))

    # Use multiprocessing to process questions on multiple GPUs
    print(f"\nStarting parallel processing on {NUM_GPUS} GPUs...")
    print("=" * 60)

    # Create pool
    pool = mp.Pool(processes=NUM_GPUS)

    # Process each question on its designated GPU
    results = pool.starmap(
        process_single_question,
        [
            (gpu_id, q_idx, q_text, MODEL_PATH, MAX_NEW_TOKENS, OUTPUT_DIR)
            for gpu_id, (q_idx, q_text) in enumerate(question_data)
        ]
    )

    # Close pool
    pool.close()
    pool.join()

    # Merge results from all GPUs
    print("\n" + "=" * 60)
    print("Merging results from all GPUs...")
    all_results = []
    for result_list in results:
        all_results.extend(result_list)

    # Save results
    save_results(all_results, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("All Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
