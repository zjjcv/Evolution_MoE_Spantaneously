"""
Ablation Study on Llama-3.1-8B-Instruct using syn_ratio_rank derived from pairwise PhiID data.

This script performs ablation experiments on attention heads:
1. Low-to-high rank ablation (redundant first)
2. High-to-low rank ablation (synergistic first)
3. Random ablation (twice for error bars)

For each strategy, we ablate heads incrementally and measure GSM8K accuracy.
Head ranking is computed by aggregating pairwise syn/red scores into per-head metrics.
"""

import os
import pandas as pd
import numpy as np
import torch
import json
import random
from tqdm import tqdm
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Configuration
MODEL_PATH = "/root/data1/zjj/Neurlps2026/Checkpoints/Meta-Llama-3.1-8B-Instruct"
PAIRWISE_CSV = "/root/data1/zjj/Neurlps2026/Results/Llama-3.1-8B-Instruct/data/pairwise/al_syn_red_pairwise.csv"
GSM8K_DATA_DIR = "/root/data1/zjj/Neurlps2026/Dataset/gsm8k"
OUTPUT_DIR = "/root/data1/zjj/Neurlps2026/Results/Llama-3.1-8B-Instruct/ablation"
NUM_QUESTIONS = 100
MAX_NEW_TOKENS = 512
RANDOM_RUNS = 2
BATCH_SIZE = 4

# Llama-3.1-8B-Instruct architecture
NUM_LAYERS = 32
NUM_HEADS = 32
TOTAL_HEADS = NUM_LAYERS * NUM_HEADS  # 1024

# Checkpoint directory for resuming
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_syn_red_data(pairwise_csv: str) -> pd.DataFrame:
    """Load pairwise PhiID data and aggregate to per-head syn_ratio_rank.

    For each head, average its syn/red scores across all pairwise interactions,
    then compute syn_ratio = Syn / (Syn + Red) and rank.
    """
    df = pd.read_csv(pairwise_csv)
    print(f"  Loaded pairwise data: {len(df):,} rows, {df['question_id'].nunique()} questions")

    # Each head appears as both (layer_1, head_1) and (layer_2, head_2)
    # Collect all syn/red contributions for each head
    head_syn = {}
    head_red = {}
    head_count = {}

    for _, row in df.iterrows():
        for lid, hid in [(row['layer_1'], row['head_1']), (row['layer_2'], row['head_2'])]:
            key = (int(lid), int(hid))
            head_syn[key] = head_syn.get(key, 0.0) + row['syn']
            head_red[key] = head_red.get(key, 0.0) + row['red']
            head_count[key] = head_count.get(key, 0) + 1

    # Average and build DataFrame
    records = []
    for (layer, head), count in head_count.items():
        avg_syn = head_syn[(layer, head)] / count
        avg_red = head_red[(layer, head)] / count
        records.append({
            'Layer': layer,
            'Head': head,
            'Syn': avg_syn,
            'Red': avg_red,
        })

    result = pd.DataFrame(records)

    # Compute syn_ratio and rank
    result['syn_ratio'] = result['Syn'] / (result['Syn'] + result['Red'])
    result['syn_ratio_rank'] = result['syn_ratio'].rank(method='dense')
    result['head_uid'] = result['Layer'].astype(str) + '_' + result['Head'].astype(str)

    result = result.sort_values(by=['Layer', 'Head']).reset_index(drop=True)
    print(f"  Aggregated to {len(result)} heads")
    print(f"  syn_ratio range: [{result['syn_ratio'].min():.4f}, {result['syn_ratio'].max():.4f}]")

    return result


def load_gsm8k_samples(data_dir: str, num_samples: int = 100) -> List[Dict]:
    """Load GSM8K dataset samples."""
    print("Loading GSM8K dataset...")
    test_file = os.path.join(data_dir, "json", "test.json")

    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    indices = np.random.choice(len(data), min(num_samples, len(data)), replace=False)
    samples = [data[int(i)] for i in indices]
    print(f"Loaded {len(samples)} questions from GSM8K")
    return samples


def get_llama_attn(model, layer_idx: int):
    """Get the self_attn module for a specific layer of Llama."""
    # Llama: model.model.layers[layer_idx].self_attn
    return model.model.layers[layer_idx].self_attn


def ablate_head_qproj_oproj(model, layer_idx: int, head_idx: int):
    """Ablate a specific attention head by zeroing out its q_proj and o_proj weights."""
    attn = get_llama_attn(model, layer_idx)

    q_proj = attn.q_proj
    o_proj = attn.o_proj

    # Llama-3.1-8B-Instruct: hidden_size=4096, num_heads=32, head_dim=128
    head_dim = q_proj.weight.shape[0] // NUM_HEADS

    with torch.no_grad():
        q_proj.weight[head_idx * head_dim:(head_idx + 1) * head_dim, :] = 0
        o_proj.weight[:, head_idx * head_dim:(head_idx + 1) * head_dim] = 0


def save_original_weights(model, layer_indices: List[int]) -> Dict:
    """Save original weights for specified layers."""
    original_weights = {}
    with torch.no_grad():
        for layer_idx in layer_indices:
            attn = get_llama_attn(model, layer_idx)
            original_weights[(layer_idx, 'q_proj')] = attn.q_proj.weight.clone()
            original_weights[(layer_idx, 'o_proj')] = attn.o_proj.weight.clone()
    return original_weights


def reset_model_weights(model, original_weights: Dict):
    """Reset model weights to original state."""
    with torch.no_grad():
        for key, tensor in original_weights.items():
            layer_idx, proj_type = key
            attn = get_llama_attn(model, layer_idx)
            if proj_type == 'q_proj':
                attn.q_proj.weight.copy_(tensor)
            elif proj_type == 'o_proj':
                attn.o_proj.weight.copy_(tensor)


def format_llama_prompt(question: str) -> str:
    """Format question for Llama-3.1-8B-Instruct."""
    msgs = [{'role': 'user', 'content': question}]
    # Note: tokenizer not available here, so use manual template
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"


def extract_gsm8k_answer(text: str) -> str:
    """Extract final answer from GSM8K output."""
    if "####" in text:
        return text.split("####")[-1].strip()
    import re
    numbers = re.findall(r'\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    return text.strip().split()[-1] if text.strip() else ""


def evaluate_accuracy(model, tokenizer, samples: List[Dict], batch_size: int = 4) -> float:
    """Evaluate model accuracy on GSM8K samples."""
    model.eval()
    correct = 0
    total = len(samples)

    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i + batch_size]
        prompts = [format_llama_prompt(s['question']) for s in batch_samples]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, output in enumerate(outputs):
            generated_text = tokenizer.decode(output[inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            predicted_answer = extract_gsm8k_answer(generated_text)
            correct_answer = extract_gsm8k_answer(batch_samples[j]['answer'])

            if predicted_answer and correct_answer:
                pred_norm = predicted_answer.replace(',', '').replace(' ', '')
                ans_norm = correct_answer.replace(',', '').replace(' ', '')
                if pred_norm == ans_norm:
                    correct += 1

    return correct / total


def run_ablation_experiment(model, tokenizer, syn_df: pd.DataFrame, samples: List[Dict],
                            strategy: str = "low_to_high", num_steps: int = 40) -> Dict:
    """Run ablation experiment with incremental head removal."""
    results = {
        'strategy': strategy,
        'num_ablated': [],
        'accuracy': []
    }

    # Sort heads by syn_ratio_rank
    if strategy == "low_to_high":
        heads = syn_df.sort_values('syn_ratio_rank', ascending=True)
    elif strategy == "high_to_low":
        heads = syn_df.sort_values('syn_ratio_rank', ascending=False)
    else:
        heads = syn_df.sample(frac=1, random_state=RANDOM_SEED)

    head_uids = heads['head_uid'].tolist()
    total_heads = len(head_uids)
    step_size = max(1, total_heads // num_steps)

    # Save original weights
    layer_indices = syn_df['Layer'].unique().tolist()
    original_weights = save_original_weights(model, layer_indices)

    print(f"\nRunning {strategy} ablation experiment...")

    num_ablated = 0
    for step in tqdm(range(num_steps + 1), desc=f"Ablation ({strategy})"):
        if step > 0:
            heads_to_ablate_count = min(step_size, total_heads - num_ablated)
            if heads_to_ablate_count <= 0:
                break

            start_idx = num_ablated
            end_idx = num_ablated + heads_to_ablate_count
            heads_to_ablate = head_uids[start_idx:end_idx]

            for head_uid in heads_to_ablate:
                layer, head = map(int, head_uid.split('_'))
                ablate_head_qproj_oproj(model, layer, head)

            num_ablated += heads_to_ablate_count

        accuracy = evaluate_accuracy(model, tokenizer, samples)
        results['num_ablated'].append(num_ablated)
        results['accuracy'].append(accuracy)

        print(f"Step {step}: Ablated {num_ablated}/{total_heads} heads, Accuracy: {accuracy:.3f}")

    reset_model_weights(model, original_weights)
    print("Model weights reset")

    return results


def plot_ablation_results(results_df: pd.DataFrame, output_dir: str):
    """Plot ablation results with error bars for random runs."""
    sns.set_theme(style="white")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Noto Serif', 'DejaVu Serif', 'Times New Roman', 'Liberation Serif'],
        'font.size': 18,
        'axes.labelsize': 24,
        'axes.titlesize': 26,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 18,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.08,
        'axes.unicode_minus': False,
        'axes.linewidth': 1.5,
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    # Collect random run data first
    random_runs = [r for r in results_df['strategy'].unique() if 'random' in r]
    random_data = results_df[results_df['strategy'].isin(random_runs)]

    if len(random_runs) > 0:
        random_stats = random_data.groupby('num_ablated').agg({
            'accuracy': ['mean', 'std']
        }).reset_index()
        random_stats.columns = ['num_ablated', 'mean', 'std']
        random_stats['std'] = random_stats['std'].fillna(0)

        ax.plot(random_stats['num_ablated'], random_stats['mean'],
                label='Random', marker='o', color='gray', alpha=0.7)
        ax.fill_between(random_stats['num_ablated'],
                        random_stats['mean'] - random_stats['std'],
                        random_stats['mean'] + random_stats['std'],
                        color='gray', alpha=0.2)

    # Plot deterministic strategies
    for strategy in results_df['strategy'].unique():
        if 'random' in strategy:
            continue
        strategy_data = results_df[results_df['strategy'] == strategy]

        if strategy == "low_to_high":
            label = "Redundant First (Low Rank)"
            color = "#d62728"
        else:
            label = "Synergistic First (High Rank)"
            color = "#1f77b4"

        ax.plot(strategy_data['num_ablated'], strategy_data['accuracy'],
                label=label, marker='o', color=color, linewidth=2)

    ax.set_xlabel('Number of Ablated Attention Heads')
    ax.set_ylabel('GSM8K Accuracy')
    ax.set_title('Attention Head Ablation Study on Llama-3.1-8B-Instruct', pad=15)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlim(0, results_df['num_ablated'].max())
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "ablation_results.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved to {plot_path}")
    plt.close()


def main():
    """Main execution function."""
    print("=" * 60)
    print("Llama-3.1-8B-Instruct Ablation Study")
    print("=" * 60)

    # 1. Load pairwise data and compute per-head ranking
    print("\n1. Computing per-head syn_ratio_rank from pairwise data...")
    syn_df = load_syn_red_data(PAIRWISE_CSV)
    print(f"   Total heads: {len(syn_df)}")
    print(f"   Layers: {syn_df['Layer'].nunique()}")
    print(f"   Heads per layer: {syn_df['Head'].nunique()}")

    # 2. Load GSM8K samples
    print("\n2. Loading GSM8K samples...")
    samples = load_gsm8k_samples(GSM8K_DATA_DIR, NUM_QUESTIONS)

    # 3. Load model
    print("\n3. Loading Llama-3.1-8B-Instruct model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"   Model loaded")

    # 4. Run ablation experiments
    print("\n4. Running ablation experiments...")
    all_results = []

    # Strategy 1: Low to high rank (redundant first)
    results_low_high = run_ablation_experiment(
        model, tokenizer, syn_df, samples,
        strategy="low_to_high",
        num_steps=40
    )
    all_results.append(results_low_high)

    # Strategy 2: High to low rank (synergistic first)
    results_high_low = run_ablation_experiment(
        model, tokenizer, syn_df, samples,
        strategy="high_to_low",
        num_steps=40
    )
    all_results.append(results_high_low)

    # Strategy 3: Random ablation (run twice)
    for run_idx in range(RANDOM_RUNS):
        results_random = run_ablation_experiment(
            model, tokenizer, syn_df, samples,
            strategy=f"random_run{run_idx + 1}",
            num_steps=40
        )
        all_results.append(results_random)

    # 5. Save results
    print("\n5. Saving results...")
    results_df = pd.DataFrame()
    for result in all_results:
        df = pd.DataFrame({
            'num_ablated': result['num_ablated'],
            'accuracy': result['accuracy'],
            'strategy': result['strategy']
        })
        results_df = pd.concat([results_df, df], ignore_index=True)

    csv_path = os.path.join(OUTPUT_DIR, "ablation_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"   Results saved to {csv_path}")

    # 6. Plot
    print("\n6. Plotting results...")
    plot_ablation_results(results_df, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Ablation study complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
