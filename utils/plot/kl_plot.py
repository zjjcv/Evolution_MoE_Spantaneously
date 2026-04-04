"""
Compute KL Divergence between each layer and the last layer.

This script reads al_plus_ml.csv data and computes the KL divergence
between each layer's hidden state and the last layer's hidden state.

Input: /data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/gsm8k/2048_length/residual_stream/al_plus_ml.csv

Output:
    - CSV with KL divergence values: /data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/gsm8k/2048_length/layer_relative_change/kl.csv
    - KL divergence plot: /data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/gsm8k/plots/kl.png

Usage:
    python utils/plot/kl_plot.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Configuration
INPUT_FILE = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/gsm8k/2048_length/residual_stream/al_plus_ml.csv"
OUTPUT_CSV = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/gsm8k/2048_length/layer_relative_change/kl.csv"
OUTPUT_PLOT = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/gsm8k/plots/kl.png"

NUM_LAYERS = 36
HIDDEN_SIZE = 4096


def load_layer_vectors(file_path: str) -> dict:
    """Load layer vectors from CSV file.

    Args:
        file_path: Path to CSV file with columns: question_id, layer, dim_0, dim_1, ...

    Returns:
        Dictionary mapping (question_id, layer) -> vector
    """
    print(f"Loading vectors from {file_path}...")

    df = pd.read_csv(file_path)
    dim_cols = [col for col in df.columns if col.startswith('dim_')]

    data_dict = {}
    for _, row in df.iterrows():
        question_id = int(row['question_id'])
        layer = int(row['layer'])
        vector = row[dim_cols].values.astype(np.float32)
        data_dict[(question_id, layer)] = vector

    print(f"  Loaded {len(data_dict)} vectors from {len(set(k[0] for k in data_dict))} questions")
    return data_dict


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax of a vector.

    Args:
        x: Input vector

    Returns:
        Softmax-normalized probability distribution
    """
    # Shift for numerical stability
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """Compute KL divergence D(P || Q).

    Args:
        p: True distribution (probability vector)
        q: Approximate distribution (probability vector)
        epsilon: Small value to avoid log(0)

    Returns:
        KL divergence value
    """
    # Add epsilon to avoid division by zero and log(0)
    p = p + epsilon
    q = q + epsilon

    # Renormalize
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Compute KL divergence
    kl = np.sum(p * np.log(p / q))
    return kl


def compute_kl_divergence_to_last_layer(vectors: dict, num_layers: int) -> pd.DataFrame:
    """Compute KL divergence between each layer and the last layer.

    Args:
        vectors: Dict[(question_id, layer)] -> vector
        num_layers: Number of layers

    Returns:
        DataFrame with columns: question_id, layer, kl_divergence
    """
    print("\nComputing KL divergence to last layer...")

    results = []
    questions = sorted(set(k[0] for k in vectors.keys()))

    for question_id in questions:
        # Get last layer vector
        last_layer = num_layers - 1
        if (question_id, last_layer) not in vectors:
            print(f"  Warning: Question {question_id} missing last layer data")
            continue

        last_vector = vectors[(question_id, last_layer)]
        # Normalize to probability distribution using softmax
        last_dist = softmax(last_vector)

        # Compute KL divergence for each layer
        for layer in range(num_layers):
            if (question_id, layer) not in vectors:
                continue

            if layer == last_layer:
                # KL divergence of last layer with itself is 0
                kl = 0.0
            else:
                vector = vectors[(question_id, layer)]
                # Normalize to probability distribution
                layer_dist = softmax(vector)

                # Compute KL divergence
                kl = kl_divergence(layer_dist, last_dist)

            results.append({
                'question_id': question_id,
                'layer': layer,
                'kl_divergence': kl
            })

    df = pd.DataFrame(results)
    print(f"  Computed KL divergence for {len(df)} (question, layer) pairs")

    return df


def aggregate_by_layer(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate KL divergence by layer (average across questions).

    Args:
        df: DataFrame with question_id, layer, kl_divergence columns

    Returns:
        DataFrame with layer and mean_kl, std_kl
    """
    print("\nAggregating by layer...")

    layer_stats = df.groupby('layer')['kl_divergence'].agg(['mean', 'std', 'count']).reset_index()
    layer_stats.columns = ['layer', 'mean_kl', 'std_kl', 'count']

    return layer_stats


def plot_kl_divergence(layer_stats: pd.DataFrame, output_path: str):
    """Plot KL divergence bar chart.

    Args:
        layer_stats: Aggregated KL divergence stats per layer
        output_path: Path to save the plot
    """
    print("\nPlotting KL divergence...")

    fig, ax = plt.subplots(1, 1, figsize=(14, 7))

    x = layer_stats['layer'].values
    y = layer_stats['mean_kl'].values

    # Create bar plot
    bars = ax.bar(x, y, color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)

    # Add value labels on top of bars
    for i, (bar, val) in enumerate(zip(bars, y)):
        if i % 2 == 0:  # Show every other label to avoid overcrowding
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.2f}',
                   ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Layer Index', fontsize=13)
    ax.set_ylabel('KL Divergence to Last Layer', fontsize=13)
    ax.set_title('Layer-wise KL Divergence from Last Layer Hidden State\n' +
                'D_KL(Layer_i || Layer_Last)', fontsize=15, pad=15)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    """Main execution function."""
    print("=" * 60)
    print("KL Divergence Analysis: Each Layer vs Last Layer")
    print("=" * 60)
    print(f"\nInput: {INPUT_FILE}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print(f"Output Plot: {OUTPUT_PLOT}")

    # Check input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"\nError: Input file not found: {INPUT_FILE}")
        return

    # Load vectors
    print("\n" + "=" * 60)
    print("Loading layer vectors...")
    print("=" * 60)

    vectors = load_layer_vectors(INPUT_FILE)

    # Compute KL divergence
    print("\n" + "=" * 60)
    print("Computing KL divergence...")
    print("=" * 60)

    kl_df = compute_kl_divergence_to_last_layer(vectors, NUM_LAYERS)

    # Save raw results
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    kl_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved raw KL divergence data to: {OUTPUT_CSV}")

    # Aggregate by layer
    layer_stats = aggregate_by_layer(kl_df)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"\nKL Divergence (mean over questions):")
    print(f"  Min: {layer_stats['mean_kl'].min():.4f} (layer {layer_stats.loc[layer_stats['mean_kl'].idxmin(), 'layer']})")
    print(f"  Max: {layer_stats['mean_kl'].max():.4f} (layer {layer_stats.loc[layer_stats['mean_kl'].idxmax(), 'layer']})")
    print(f"  Mean: {layer_stats['mean_kl'].mean():.4f}")
    print(f"  Median: {layer_stats['mean_kl'].median():.4f}")

    # Save aggregated results
    agg_output = OUTPUT_CSV.replace('.csv', '_aggregated.csv')
    layer_stats.to_csv(agg_output, index=False)
    print(f"\nSaved aggregated KL divergence data to: {agg_output}")

    # Plot
    plot_kl_divergence(layer_stats, OUTPUT_PLOT)

    print("\n" + "=" * 60)
    print("All Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
