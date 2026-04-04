"""
Head ablation experiment based on Syn-Red Rank for Qwen3-8B-Base.

Compares three ablation strategies:
1. High-to-low Syn-Red Rank (ablate most synergistic-redundant first)
2. Low-to-high Syn-Red Rank (ablate least synergistic-redundant first)
3. Random ablation (repeated 3 times for statistical robustness)

Uses 8 GPUs for parallel inference.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from multiprocessing import Process, Queue
import traceback
from datetime import datetime

# Configuration
MODEL_PATH = "/data/zjj/Synergistic_Core/Qwen-3-8B-base"
GSM8K_DATA_DIR = "/data/zjj/Synergistic_Core/data/gsm8k"
SYN_RED_RANK_PATH = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/gsm8k/gsm8k_al_syn_red_rank.csv"
OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/simple_ablation"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "head_ablation.csv")

NUM_QUESTIONS = 200
NUM_GPUS = 8
ABLATION_PERCENT = 0.01  # Ablate 1% of heads each step
HEADS_PER_LAYER = 32
NUM_LAYERS = 36
TOTAL_HEADS = NUM_LAYERS * HEADS_PER_LAYER  # 1152
HEADS_PER_STEP = int(TOTAL_HEADS * ABLATION_PERCENT)  # ~11-12 heads
NUM_RANDOM_REPEATS = 3  # Number of random ablation repetitions

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_gsm8k_samples(num_samples: int = 200):
    """Load GSM8K dataset samples from local JSON file."""
    test_file = os.path.join(GSM8K_DATA_DIR, "json", "test.json")

    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = data[:num_samples]

    print(f"Loaded {len(samples)} questions from GSM8K")
    return samples


def load_syn_red_rank(csv_path: str):
    """Load and sort attention heads by Syn-Red Rank."""
    df = pd.read_csv(csv_path)

    # Ensure Layer and Head are integers
    df['Layer'] = df['Layer'].astype(int)
    df['Head'] = df['Head'].astype(int)

    print(f"Loaded {len(df)} attention heads")
    print(f"Layers: {df['Layer'].nunique()}")
    print(f"Syn_Red_Rank range: [{df['Syn_Red_Rank'].min():.1f}, {df['Syn_Red_Rank'].max():.1f}]")

    return df


def format_prompt(question: str):
    """Format question with few-shot prompting."""
    return f"""Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
A: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.
#### 18

Q: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
A: It takes 2/2 = <<2/2=1>>1 bolt of white fiber.
So the total amount of fabric is 2 + 1 = <<2+1=3>>3 bolts of fabric.
#### 3

Q: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?
A: The value of the house increased by 80,000 * 1.5 = $<<80,000*1.5=120,000>>120,000.
So the new value of the house is 80,000 + 120,000 = $<<80,000+120,000=200,000>>200,000.
The profit is 200,000 - 80,000 - 50,000 = $<<200,000-80,000-50,000=70,000>>70,000.
#### 70,000

Q: {question}
A:"""


def extract_final_answer(text: str):
    """Extract final answer from GSM8K format."""
    import re
    if "####" in text:
        after_hash = text.split("####")[-1].strip()
        if after_hash and len(after_hash.split()) > 0:
            return after_hash.split()[0]
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    return numbers[-1] if numbers else ""


def ablate_heads(model, heads_to_ablate: list):
    """Ablate specified attention heads by setting their q_proj and o_proj weights to zero."""
    head_dim = model.config.hidden_size // model.config.num_attention_heads  # 128

    with torch.no_grad():
        for layer_idx, head_idx in heads_to_ablate:
            layer = model.model.layers[layer_idx].self_attn
            layer.q_proj.weight[head_idx * head_dim:(head_idx + 1) * head_dim, :] = 0
            layer.o_proj.weight[:, head_idx * head_dim:(head_idx + 1) * head_dim] = 0


def evaluate_on_gpu(gpu_id: int, model_path: str, samples: list, heads_to_ablate: list,
                    result_queue: Queue, batch_size: int = 8):
    """Evaluate model on a specific GPU with specified heads ablated."""
    try:
        device = f"cuda:{gpu_id}"

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map={"": device},
            trust_remote_code=True
        )
        model.eval()

        if heads_to_ablate:
            ablate_heads(model, heads_to_ablate)

        correct = 0
        total = len(samples)

        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i + batch_size]
            prompts = [format_prompt(s['question']) for s in batch_samples]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            for j, output in enumerate(outputs):
                generated = tokenizer.decode(
                    output[inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                predicted = extract_final_answer(generated)
                expected = extract_final_answer(batch_samples[j]['answer'])

                if predicted and expected:
                    pred_clean = predicted.replace(',', '').replace('$', '').replace(' ', '')
                    exp_clean = expected.replace(',', '').replace('$', '').replace(' ', '')
                    try:
                        pred_norm = str(int(float(pred_clean)))
                        exp_norm = str(int(float(exp_clean)))
                        if pred_norm == exp_norm and pred_norm != "" and pred_norm != "inf":
                            correct += 1
                    except (ValueError, TypeError, OverflowError):
                        pass

        accuracy = correct / total
        result_queue.put((gpu_id, accuracy, correct, total))

        del model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"[GPU {gpu_id}] Error: {e}")
        traceback.print_exc()
        result_queue.put((gpu_id, 0.0, 0, len(samples)))


def parallel_evaluate(model_path: str, samples: list, heads_to_ablate: list, num_gpus: int = 8):
    """Run evaluation in parallel on multiple GPUs."""
    samples_per_gpu = len(samples) // num_gpus
    gpu_samples = []
    for i in range(num_gpus):
        start = i * samples_per_gpu
        end = start + samples_per_gpu if i < num_gpus - 1 else len(samples)
        gpu_samples.append(samples[start:end])

    result_queue = Queue()
    processes = []

    for gpu_id in range(num_gpus):
        p = Process(
            target=evaluate_on_gpu,
            args=(gpu_id, model_path, gpu_samples[gpu_id], heads_to_ablate, result_queue)
        )
        p.start()
        processes.append(p)

    results = []
    for _ in range(num_gpus):
        results.append(result_queue.get())

    for p in processes:
        p.join()

    total_correct = sum(r[2] for r in results)
    total_samples = sum(r[3] for r in results)
    overall_accuracy = total_correct / total_samples

    return overall_accuracy, total_correct, total_samples


def run_ablation_experiment(experiment_type: str, heads_ranked: list, samples: list,
                            repeat_idx: int = 0, syn_red_df: pd.DataFrame = None):
    """Run a single ablation experiment with given head ordering.

    Args:
        experiment_type: 'high_to_low', 'low_to_high', or 'random'
        heads_ranked: List of (layer, head) tuples in ablation order
        samples: Test samples
        repeat_idx: Repeat index for random experiments
        syn_red_df: Syn-Red rank dataframe (for logging)

    Returns:
        List of result dictionaries
    """
    print("\n" + "=" * 60)
    print(f"Experiment: {experiment_type}")
    if repeat_idx > 0:
        print(f"Repeat: {repeat_idx}/{NUM_RANDOM_REPEATS}")
    print("=" * 60)

    num_steps = (len(heads_ranked) + HEADS_PER_STEP - 1) // HEADS_PER_STEP
    results_list = []

    # Progressive ablation
    for step in range(num_steps):
        start_idx = step * HEADS_PER_STEP
        end_idx = min(start_idx + HEADS_PER_STEP, len(heads_ranked))
        cumulative_ablated = heads_ranked[:end_idx]

        print(f"\nStep {step + 1}/{num_steps}: Cumulative ablated {len(cumulative_ablated)} heads")

        if syn_red_df is not None and step == 0:
            print(f"  Top 5 heads being ablated:")
            for layer, head in cumulative_ablated[:5]:
                row = syn_red_df[(syn_red_df['Layer'] == layer) & (syn_red_df['Head'] == head)]
                if len(row) > 0:
                    print(f"    Layer {layer:2d} Head {head:2d} (Syn-Red Rank: {row['Syn_Red_Rank'].values[0]:.1f})")

        acc, correct, total = parallel_evaluate(
            MODEL_PATH, samples, heads_to_ablate=cumulative_ablated, num_gpus=NUM_GPUS
        )

        print(f"  Accuracy: {acc:.4f} ({correct}/{total})")

        results_list.append({
            'experiment_type': experiment_type,
            'repeat_idx': repeat_idx,
            'ablation_step': step + 1,
            'heads_ablated_cumulative': len(cumulative_ablated),
            'pct_ablated': len(cumulative_ablated) / TOTAL_HEADS * 100,
            'accuracy': acc,
            'correct': correct,
            'total': total
        })

        # Save intermediate results
        df = pd.DataFrame(results_list)
        df.to_csv(OUTPUT_CSV, index=False)

    return results_list


def main():
    print("=" * 60)
    print("Head Ablation Experiment: Multiple Strategies")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Questions: {NUM_QUESTIONS}")
    print(f"Total heads: {TOTAL_HEADS}")
    print(f"Heads per ablation step: {HEADS_PER_STEP} ({ABLATION_PERCENT*100}%)")
    print(f"GPUs: {NUM_GPUS}")
    print(f"Random repeats: {NUM_RANDOM_REPEATS}")

    # Load data
    print("\n1. Loading data...")
    samples = load_gsm8k_samples(NUM_QUESTIONS)
    syn_red_df = load_syn_red_rank(SYN_RED_RANK_PATH)

    # Calculate number of ablation steps
    num_steps = (TOTAL_HEADS + HEADS_PER_STEP - 1) // HEADS_PER_STEP
    print(f"\nTotal ablation steps: {num_steps}")

    # Results storage
    all_results = []

    # ===== Baseline (no ablation) =====
    print("\n" + "=" * 60)
    print("Evaluating baseline (no ablation)...")
    print("=" * 60)
    baseline_acc, correct, total = parallel_evaluate(
        MODEL_PATH, samples, heads_to_ablate=[], num_gpus=NUM_GPUS
    )
    print(f"Baseline accuracy: {baseline_acc:.4f} ({correct}/{total})")

    # ===== Experiment 1: High-to-Low Syn-Red Rank =====
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: High-to-Low Syn-Red Rank Ablation")
    print("=" * 60)
    print("Ablating heads with HIGHEST Syn-Red Rank first")

    df_sorted_high_to_low = syn_red_df.sort_values('Syn_Red_Rank', ascending=False).reset_index(drop=True)
    heads_high_to_low = list(zip(df_sorted_high_to_low['Layer'].values, df_sorted_high_to_low['Head'].values))

    results_h2l = run_ablation_experiment(
        'high_to_low',
        heads_high_to_low,
        samples,
        syn_red_df=df_sorted_high_to_low
    )
    all_results.extend(results_h2l)

    # ===== Experiment 2: Low-to-High Syn-Red Rank =====
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Low-to-High Syn-Red Rank Ablation")
    print("=" * 60)
    print("Ablating heads with LOWEST Syn-Red Rank first")

    df_sorted_low_to_high = syn_red_df.sort_values('Syn_Red_Rank', ascending=True).reset_index(drop=True)
    heads_low_to_high = list(zip(df_sorted_low_to_high['Layer'].values, df_sorted_low_to_high['Head'].values))

    results_l2h = run_ablation_experiment(
        'low_to_high',
        heads_low_to_high,
        samples,
        syn_red_df=df_sorted_low_to_high
    )
    all_results.extend(results_l2h)

    # ===== Experiment 3: Random Ablation (3 repeats) =====
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Random Ablation (3 repeats)")
    print("=" * 60)

    np.random.seed(42)  # For reproducibility

    for repeat in range(1, NUM_RANDOM_REPEATS + 1):
        print(f"\n--- Random Ablation: Repeat {repeat}/{NUM_RANDOM_REPEATS} ---")

        # Generate random ordering
        all_heads = list(zip(syn_red_df['Layer'].values, syn_red_df['Head'].values))
        np.random.shuffle(all_heads)

        results_random = run_ablation_experiment(
            'random',
            all_heads,
            samples,
            repeat_idx=repeat
        )
        all_results.extend(results_random)

    # ===== Final Analysis =====
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    # Save all results
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nAll results saved to: {OUTPUT_CSV}")

    # Summary statistics by experiment type
    print("\n" + "=" * 60)
    print("Accuracy at Different Ablation Levels")
    print("=" * 60)

    # Create summary table
    ablation_levels = [0, 10, 20, 30, 40, 50]  # Percentage of heads ablated

    summary_data = []
    for level in ablation_levels:
        row = {'Ablation %': f'{level}%'}
        for exp_type in ['high_to_low', 'low_to_high', 'random']:
            if level == 0:
                # Baseline
                row[f'Baseline'] = f'{baseline_acc:.3f}'
            else:
                # Find closest ablation level
                exp_df = df[(df['experiment_type'] == exp_type) &
                           (df['pct_ablated'] >= level - 1) &
                           (df['pct_ablated'] <= level + 1)]
                if len(exp_df) > 0:
                    if exp_type == 'random':
                        # Average over repeats
                        acc = exp_df['accuracy'].mean()
                        row[f'{exp_type}_avg'] = f'{acc:.3f}'
                        # Also show std
                        std = exp_df['accuracy'].std()
                        row[f'{exp_type}_std'] = f'±{std:.3f}'
                    else:
                        acc = exp_df['accuracy'].values[0]
                        row[f'{exp_type}'] = f'{acc:.3f}'
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    print("\nSummary Table:")
    print(summary_df.to_string(index=False))

    # Comparison at key ablation levels
    print("\n" + "=" * 60)
    print("Key Comparisons")
    print("=" * 60)

    key_levels = [10, 20, 30, 50]

    for level in key_levels:
        print(f"\nAt {level}% ablation:")

        h2l_acc = df[(df['experiment_type'] == 'high_to_low') &
                    (df['pct_ablated'] >= level - 1) &
                    (df['pct_ablated'] <= level + 1)]['accuracy'].values
        l2h_acc = df[(df['experiment_type'] == 'low_to_high') &
                    (df['pct_ablated'] >= level - 1) &
                    (df['pct_ablated'] <= level + 1)]['accuracy'].values
        rnd_acc = df[(df['experiment_type'] == 'random') &
                    (df['pct_ablated'] >= level - 1) &
                    (df['pct_ablated'] <= level + 1)]['accuracy']

        if len(h2l_acc) > 0:
            print(f"  High-to-Low: {h2l_acc[0]:.4f}")
        if len(l2h_acc) > 0:
            print(f"  Low-to-High: {l2h_acc[0]:.4f}")
        if len(rnd_acc) > 0:
            print(f"  Random (mean±std): {rnd_acc.mean():.4f} ± {rnd_acc.std():.4f}")

    # Final comparison
    print("\n" + "=" * 60)
    print("Conclusion")
    print("=" * 60)

    final_h2l = df[df['experiment_type'] == 'high_to_low']['accuracy'].values[-1]
    final_l2h = df[df['experiment_type'] == 'low_to_high']['accuracy'].values[-1]
    final_rnd = df[df['experiment_type'] == 'random']['accuracy'].mean()

    print(f"Final accuracy (all heads ablated except ~0%):")
    print(f"  High-to-Low Syn-Red: {final_h2l:.4f}")
    print(f"  Low-to-High Syn-Red:  {final_l2h:.4f}")
    print(f"  Random (average):     {final_rnd:.4f}")

    if final_l2h > final_h2l:
        print("\nConclusion: Ablating LOW Syn-Red rank heads preserves accuracy better")
    else:
        print("\nConclusion: Ablating HIGH Syn-Red rank heads preserves accuracy better")


if __name__ == "__main__":
    main()
