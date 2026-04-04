"""
Compute pairwise Synergy-Redundancy for ARC dataset using multiprocessing.

Reads L2 norm time series from proxy collection, computes
PhiID-based synergy and redundancy for all pairs of attention heads/layers,
using multiprocessing for parallel computation.

Input: arc_{difficulty}_{proxy_type}.csv (L2 norm time series per head/layer)
Output: {proxy_type}_syn_red_pairwise.csv (pairwise syn/red for each question)

Usage:
    python utils/compute_al_syn_red_pairwise_mp.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations
from multiprocessing import Pool, cpu_count
from functools import partial

# Try to import PhiID library
try:
    from phyid.calculate import calc_PhiID
except ImportError:
    print("Error: phyid module not found. Please install integrated-info-decomp:")
    print("   pip install -e /path/to/integrated-info-decomp/")
    sys.exit(1)

# ============================================================
# Configuration
# ============================================================
INPUT_BASE_DIR = os.environ.get("INPUT_BASE_DIR", "/data/zjj/Synergistic_Core/results/Gemma3-4B-Instruct/data")
OUTPUT_BASE_DIR = os.environ.get("OUTPUT_BASE_DIR", "/data/zjj/Synergistic_Core/results/Gemma3-4B-Instruct/data")

# ARC difficulties
DIFFICULTIES = ['easy', 'challenge']

# Multiprocessing Configuration
N_WORKERS = 100  # Use 100 CPU cores for parallel computation
TAU = 1
KIND = "gaussian"
REDUNDANCY = "MMI"

print(f"Using {N_WORKERS} workers (out of {cpu_count()} CPU cores available)")


# ============================================================
# PhiID Computation Functions
# ============================================================

def compute_single_pair_phiid(ts1, ts2):
    """
    Compute PhiID for a single pair of time series.

    Returns:
        (syn, red) tuple or (nan, nan) if computation fails
    """
    try:
        atoms_res, _ = calc_PhiID(ts1, ts2, tau=TAU, kind=KIND, redundancy=REDUNDANCY)
        syn = float(np.nanmean(np.asarray(atoms_res["sts"])))
        red = float(np.nanmean(np.asarray(atoms_res["rtr"])))
        return (syn, red)
    except Exception as e:
        return (np.nan, np.nan)


def process_question_worker(q_idx, df_dict, effective_lengths_dict, components, proxy_type, worker_id=None):
    """
    Worker function to process a single question.

    Args:
        q_idx: Question ID
        df_dict: Dictionary with question data
        effective_lengths_dict: Dictionary with effective lengths
        components: List of component UIDs
        proxy_type: 'al' (per-head), 'ml' or 'al_plus_ml' (per-layer)
        worker_id: Worker ID for progress bar positioning

    Returns:
        List of tuples depending on proxy_type:
        - al: (layer_1, head_1, layer_2, head_2, syn, red)
        - ml/al_plus_ml: (layer_1, layer_2, syn, red)
    """
    # Get data for this question
    q_df = df_dict[q_idx]
    # Use actual step columns present in the data (not effective_length from generation)
    actual_steps = len([c for c in q_df.columns if c.startswith('step_')])
    effective_length = min(effective_lengths_dict.get(q_idx, actual_steps), actual_steps)

    step_cols = [f'step_{i+1}' for i in range(effective_length)]

    # Extract time series for each component
    ts_dict = {}
    for comp in components:
        if proxy_type == 'al':
            # For al, component is (layer, head) tuple
            layer, head = comp
            comp_df = q_df[(q_df['layer'] == layer) & (q_df['head'] == head)]
        else:
            # For ml/al_plus_ml, component is layer
            comp_df = q_df[q_df['layer'] == comp]

        if len(comp_df) > 0:
            ts = comp_df[step_cols].values.flatten().astype(np.float64)
            ts_dict[comp] = ts
        else:
            ts_dict[comp] = np.zeros(effective_length, dtype=np.float64)

    # Generate all pairs
    component_pairs = list(combinations(components, 2))
    num_pairs = len(component_pairs)

    results = []

    # Compute PhiID for each pair with progress bar
    position = worker_id if worker_id is not None else q_idx
    for comp1, comp2 in tqdm(component_pairs, total=num_pairs,
                            desc=f"Q{q_idx}",
                            position=position,
                            leave=False,
                            ncols=80):
        ts1 = ts_dict[comp1]
        ts2 = ts_dict[comp2]

        syn, red = compute_single_pair_phiid(ts1, ts2)

        if proxy_type == 'al':
            layer1, head1 = comp1
            layer2, head2 = comp2
            results.append((layer1, head1, layer2, head2, syn, red))
        else:
            # ml or al_plus_ml: components are layer indices
            results.append((comp1, comp2, syn, red))

    return (q_idx, results)


def compute_pairwise_for_proxy_type(proxy_type: str, difficulty: str):
    """Compute pairwise syn/red for one proxy type for all questions."""
    print(f"\n{'=' * 60}")
    print(f"Processing ARC-{difficulty.capitalize()} - {proxy_type.upper()}")
    print(f"{'=' * 60}")

    # Setup paths
    difficulty_dir = os.path.join(OUTPUT_BASE_DIR, difficulty)
    os.makedirs(difficulty_dir, exist_ok=True)

    # Input file name
    input_file = os.path.join(INPUT_BASE_DIR, difficulty, f"arc_{difficulty}_{proxy_type}.csv")
    output_file = os.path.join(difficulty_dir, f"{proxy_type}_syn_red_pairwise.csv")
    effective_lengths_file = os.path.join(INPUT_BASE_DIR, difficulty, f"arc_{difficulty}_effective_lengths.csv")

    if not os.path.exists(input_file):
        print(f"Warning: Input file not found: {input_file}")
        return

    # Load input data
    print(f"Loading data from: {input_file}")
    df = pd.read_csv(input_file)

    # Load effective lengths
    if os.path.exists(effective_lengths_file):
        effective_lengths_df = pd.read_csv(effective_lengths_file)
        effective_lengths_dict = dict(zip(effective_lengths_df['question_id'], effective_lengths_df['effective_length']))
        print(f"Loaded effective lengths for {len(effective_lengths_dict)} questions")
    else:
        print("Warning: Effective lengths file not found, using all steps")
        max_steps = len([c for c in df.columns if c.startswith('step_')])
        effective_lengths_dict = {qid: max_steps for qid in df['question_id'].unique()}

    # Get unique questions and components
    question_ids = sorted(df['question_id'].unique())

    if proxy_type == 'al':
        # For al: components are (layer, head) pairs - use set to deduplicate
        components = sorted(set(zip(df['layer'], df['head'])))
    else:
        # For ml/al_plus_ml: components are layer indices
        components = sorted(df['layer'].unique())

    print(f"Questions: {len(question_ids)}")
    print(f"Components: {len(components)}")
    print(f"Pairs per question: {len(list(combinations(components, 2))):,}")

    # Prepare data dictionary for each question
    print("\nPreparing data for multiprocessing...")
    df_dict = {qid: df[df['question_id'] == q_idx].copy() for qid, q_idx in enumerate(question_ids)}

    # Prepare worker arguments
    worker_args = [(qid, df_dict, effective_lengths_dict, components, proxy_type, i % N_WORKERS)
                   for i, qid in enumerate(question_ids)]

    # Process questions in parallel
    print(f"\nProcessing {len(question_ids)} questions with {N_WORKERS} workers...")

    all_results = []

    with Pool(N_WORKERS) as pool:
        results_list = list(tqdm(
            pool.starmap(process_question_worker, worker_args),
            total=len(question_ids),
            desc="Overall progress"
        ))

    # Collect results from all workers
    for q_idx, question_results in results_list:
        for result in question_results:
            all_results.append((q_idx,) + result)

    # Save results
    print(f"\n{'='*60}")
    print("Saving results...")
    print(f"{'='*60}")

    if proxy_type == 'al':
        result_df = pd.DataFrame(all_results, columns=[
            'question_id', 'layer_1', 'head_1', 'layer_2', 'head_2', 'syn', 'red'
        ])
        result_df = result_df.sort_values(['question_id', 'layer_1', 'head_1', 'layer_2', 'head_2'])
    else:
        result_df = pd.DataFrame(all_results, columns=[
            'question_id', 'layer_1', 'layer_2', 'syn', 'red'
        ])
        result_df = result_df.sort_values(['question_id', 'layer_1', 'layer_2'])

    result_df.to_csv(output_file, index=False)

    print(f"\nResults saved to: {output_file}")
    print(f"Total records: {len(result_df):,}")

    # Summary statistics
    valid_syn = result_df['syn'].dropna()
    valid_red = result_df['red'].dropna()
    print(f"\nSummary statistics:")
    print(f"  Syn - mean: {valid_syn.mean():.6f}, std: {valid_syn.std():.6f}")
    print(f"  Red - mean: {valid_red.mean():.6f}, std: {valid_red.std():.6f}")
    print(f"  Valid pairs: {len(valid_syn):,} / {len(result_df):,}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("ARC Pairwise Syn-Red Computation (Gemma3-4B-Instruct)")
    print("=" * 60)
    print(f"\nWorkers: {N_WORKERS}")
    print(f"CPU cores: {cpu_count()}")

    # Process all proxy types for each difficulty
    proxy_types = ['al', 'ml', 'al_plus_ml']

    for difficulty in DIFFICULTIES:
        print(f"\n{'#'*60}")
        print(f"# Processing ARC-{difficulty.capitalize()}")
        print(f"{'#'*60}")

        for proxy_type in proxy_types:
            compute_pairwise_for_proxy_type(proxy_type, difficulty)

    print("\n" + "=" * 60)
    print("All Done!")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_BASE_DIR}")
    print("\nGenerated files for each difficulty (easy/challenge):")
    print("  - al_syn_red_pairwise.csv")
    print("  - ml_syn_red_pairwise.csv")
    print("  - al_plus_ml_syn_red_pairwise.csv")


if __name__ == "__main__":
    main()
