"""
Compute residual stream ratios and cosine similarities from pre-collected layer data.

This script reads the pre-collected layer vectors from layer_proxy_collection.py
and computes:
- ||al||² / ||h||², ||ml||² / ||h||², ||al+ml||² / ||h||²
- cos(al, h_l), cos(ml, h_l), cos(al+ml, h_l)

Then plots layer-wise ratios.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Global style (consistent with other plots) ─────────────────────────
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
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})

# ── Colour palette ──────────────────────────────────────────────────────
_COLOR_AL = '#E64B35'           # warm red (Nature-style)
_COLOR_ML = '#4DBBD5'           # teal / cyan
_COLOR_AL_ML = '#00A087'        # deep teal-green

# Configuration
INPUT_DIR = "/root/data1/zjj/Neurlps2026/Results/Llama-3.1-8B-Instruct/data/residual_stream"
OUTPUT_DIR = INPUT_DIR
PLOT_OUTPUT_DIR = "/root/data1/zjj/Neurlps2026/Results/Plots/Llama-3.1-8B-Instruct"

# Input files
AL_INPUT = os.path.join(INPUT_DIR, "al.csv")
ML_INPUT = os.path.join(INPUT_DIR, "ml.csv")
AL_PLUS_ML_INPUT = os.path.join(INPUT_DIR, "al_plus_ml.csv")

# Output files
AL_RATIO_OUTPUT = os.path.join(OUTPUT_DIR, "al_ratio.csv")
ML_RATIO_OUTPUT = os.path.join(OUTPUT_DIR, "ml_ratio.csv")
AL_PLUS_ML_RATIO_OUTPUT = os.path.join(OUTPUT_DIR, "al_plus_ml_ratio.csv")
AL_COS_OUTPUT = os.path.join(OUTPUT_DIR, "al_cos.csv")
ML_COS_OUTPUT = os.path.join(OUTPUT_DIR, "ml_cos.csv")
AL_PLUS_ML_COS_OUTPUT = os.path.join(OUTPUT_DIR, "al_plus_ml_cos.csv")

# Model parameters
NUM_LAYERS = 32
HIDDEN_SIZE = 4096


def load_layer_vectors(file_path):
    """Load layer vectors from CSV file.

    Args:
        file_path: Path to CSV file with columns: question_id, layer, dim_0, dim_1, ...

    Returns:
        Dictionary mapping (question_id, layer) -> vector
    """
    print(f"Loading data from {file_path}...")

    df = pd.read_csv(file_path)

    # Get dimension columns
    dim_cols = [col for col in df.columns if col.startswith('dim_')]

    # Build dictionary
    data_dict = {}
    for _, row in df.iterrows():
        question_id = int(row['question_id'])
        layer = int(row['layer'])
        vector = row[dim_cols].values.astype(np.float32)
        data_dict[(question_id, layer)] = vector

    print(f"  Loaded {len(data_dict)} vectors from {len(set(k[0] for k in data_dict))} questions")
    return data_dict


def compute_ratios_and_cosine(al_data, ml_data, al_plus_ml_data):
    """Compute residual stream ratios and cosine similarities.

    Args:
        al_data: Dict[(question_id, layer)] -> vector
        ml_data: Dict[(question_id, layer)] -> vector
        al_plus_ml_data: Dict[(question_id, layer)] -> vector

    Returns:
        Six DataFrames: al/ml/al_plus_ml ratios and cosine similarities
    """
    print("\nReconstructing h_l (residual stream input)...")
    print("Computing ratios and cosine similarities...")

    # First, reconstruct h_l (residual stream input at layer l) for each question
    # h_l = al_{l-1} + ml_{l-1} (previous layer's output)
    # h_0 = 0 (before first layer)

    # Group by question and reconstruct h_l
    questions = sorted(set(k[0] for k in al_plus_ml_data.keys()))
    max_layer = max(k[1] for k in al_plus_ml_data.keys())

    # h_data: (question_id, layer) -> h_l vector
    h_data = {}

    for q_id in questions:
        # h_0 = 0 (before first layer)
        h_data[(q_id, 0)] = np.zeros(HIDDEN_SIZE, dtype=np.float32)

        # For layer l, h_l = al_{l-1} + ml_{l-1} (previous layer's output)
        for layer in range(1, max_layer + 1):
            if (q_id, layer - 1) in al_plus_ml_data:
                h_data[(q_id, layer)] = al_plus_ml_data[(q_id, layer - 1)]
            else:
                h_data[(q_id, layer)] = np.zeros(HIDDEN_SIZE, dtype=np.float32)

    # Get all unique keys
    keys = set(al_data.keys()) & set(ml_data.keys()) & set(al_plus_ml_data.keys())
    print(f"  Processing {len(keys)} (question, layer) combinations")

    # Storage
    al_ratios, ml_ratios, al_plus_ml_ratios = [], [], []
    al_cosine, ml_cosine, al_plus_ml_cosine = [], [], []

    def cosine_similarity(v1, v2):
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        return np.dot(v1, v2) / (norm1 * norm2)

    for key in keys:
        question_id, layer = key

        al_vec = al_data[key]
        ml_vec = ml_data[key]
        al_plus_ml_vec = al_plus_ml_data[key]

        # Get h_l (residual stream input at layer l)
        h_vec = h_data[key]

        # Compute norms
        al_norm_sq = np.sum(al_vec ** 2)
        ml_norm_sq = np.sum(ml_vec ** 2)
        al_plus_ml_norm_sq = np.sum(al_plus_ml_vec ** 2)
        h_norm_sq = np.sum(h_vec ** 2)

        # Compute ratios: ||al||² / ||h||², etc.
        if h_norm_sq > 1e-10:
            al_ratio = al_norm_sq / h_norm_sq
            ml_ratio = ml_norm_sq / h_norm_sq
            al_plus_ml_ratio = al_plus_ml_norm_sq / h_norm_sq
        else:
            al_ratio = ml_ratio = al_plus_ml_ratio = 0.0

        # Compute cosine similarities
        al_cos = cosine_similarity(al_vec, h_vec)
        ml_cos = cosine_similarity(ml_vec, h_vec)
        al_plus_ml_cos = cosine_similarity(al_plus_ml_vec, h_vec)

        al_ratios.append({'question_id': question_id, 'layer': layer, 'ratio': al_ratio})
        ml_ratios.append({'question_id': question_id, 'layer': layer, 'ratio': ml_ratio})
        al_plus_ml_ratios.append({'question_id': question_id, 'layer': layer, 'ratio': al_plus_ml_ratio})

        al_cosine.append({'question_id': question_id, 'layer': layer, 'cosine': al_cos})
        ml_cosine.append({'question_id': question_id, 'layer': layer, 'cosine': ml_cos})
        al_plus_ml_cosine.append({'question_id': question_id, 'layer': layer, 'cosine': al_plus_ml_cos})

    return (pd.DataFrame(al_ratios), pd.DataFrame(ml_ratios), pd.DataFrame(al_plus_ml_ratios),
            pd.DataFrame(al_cosine), pd.DataFrame(ml_cosine), pd.DataFrame(al_plus_ml_cosine))


def aggregate_by_layer(df, value_col):
    """Aggregate by layer (average across all questions).

    Args:
        df: DataFrame with question_id, layer, value_col
        value_col: Column name to aggregate

    Returns:
        DataFrame with layer and mean_value
    """
    layer_stats = df.groupby('layer')[value_col].agg(['mean', 'std', 'count']).reset_index()
    layer_stats.columns = ['layer', 'mean_value', 'std_value', 'count']
    return layer_stats


def save_results(al_df, ml_df, al_plus_ml_df, al_cos_df, ml_cos_df, al_plus_ml_cos_df):
    """Save computed ratios and cosine similarities to CSV files."""
    print("\nSaving results...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    al_df.to_csv(AL_RATIO_OUTPUT, index=False)
    print(f"  Saved: {AL_RATIO_OUTPUT}")

    ml_df.to_csv(ML_RATIO_OUTPUT, index=False)
    print(f"  Saved: {ML_RATIO_OUTPUT}")

    al_plus_ml_df.to_csv(AL_PLUS_ML_RATIO_OUTPUT, index=False)
    print(f"  Saved: {AL_PLUS_ML_RATIO_OUTPUT}")

    al_cos_df.to_csv(AL_COS_OUTPUT, index=False)
    print(f"  Saved: {AL_COS_OUTPUT}")

    ml_cos_df.to_csv(ML_COS_OUTPUT, index=False)
    print(f"  Saved: {ML_COS_OUTPUT}")

    al_plus_ml_cos_df.to_csv(AL_PLUS_ML_COS_OUTPUT, index=False)
    print(f"  Saved: {AL_PLUS_ML_COS_OUTPUT}")


def _stacked_bar(ax, x, vals_list, colors, labels):
    """Draw stacked bars: largest at bottom, smallest on top, white separators."""
    arr = np.column_stack(vals_list)
    order = np.argsort(arr, axis=1)
    n = len(x)
    bottoms = np.zeros(n)

    for col_idx in range(arr.shape[1]):
        values = np.array([arr[i, order[i, col_idx]] for i in range(n)])
        lbl_idx = order[0, col_idx]
        if col_idx == arr.shape[1] - 1:
            ax.bar(x, values, width=0.88, bottom=bottoms,
                   color=colors[lbl_idx], label=labels[lbl_idx],
                   edgecolor='white', linewidth=0.8)
        else:
            ax.bar(x, values, width=0.88, bottom=bottoms,
                   color=colors[lbl_idx],
                   edgecolor='white', linewidth=0.8)
        bottoms += values

    return bottoms


def _setup_axes(ax, n_layers, xlabel, ylabel, legend_labels, legend_colors):
    """Common axis styling for both plots."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.8, n_layers - 0.2)
    ax.set_xticks(range(0, n_layers, 2))
    ax.grid(True, axis='y', linestyle='-', alpha=0.10, linewidth=0.5, color='#666666')
    ax.tick_params(axis='both', length=5)

    leg = ax.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, fc=c, ec='none', alpha=0.92)
                 for c in legend_colors],
        labels=legend_labels,
        frameon=True, fancybox=False, edgecolor='#cccccc',
        framealpha=0.95, loc='upper center', ncol=3,
        bbox_to_anchor=(0.5, 1.02),
        handletextpad=0.5, columnspacing=1.8, borderpad=0.6,
        handlelength=1.8, handleheight=1.0,
    )
    leg.get_frame().set_linewidth(0.8)


def plot_ratios(al_stats, ml_stats, al_plus_ml_stats):
    """Plot layer-wise residual stream ratios (stacked)."""
    print("\nPlotting residual stream ratios...")

    n_layers = len(al_stats)
    fig_width = max(16, n_layers * 0.52)
    fig_height = 7.0

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    x = np.arange(n_layers)
    vals = [al_stats['mean_value'].values,
            ml_stats['mean_value'].values,
            al_plus_ml_stats['mean_value'].values]
    colors = [_COLOR_AL, _COLOR_ML, _COLOR_AL_ML]
    labels = [r'$||\mathrm{AL}||^2 / ||h||^2$',
              r'$||\mathrm{ML}||^2 / ||h||^2$',
              r'$||\mathrm{AL+ML}||^2 / ||h||^2$']

    _stacked_bar(ax, x, vals, colors, labels)
    _setup_axes(ax, n_layers, 'Layer', 'Ratio', labels, colors)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    output_path = os.path.join(PLOT_OUTPUT_DIR, "residual_stream_ratios.png")
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    plt.savefig(output_path, pad_inches=0.12)
    print(f"  Saved: {output_path}")
    plt.close()


def plot_cosine(al_stats, ml_stats, al_plus_ml_stats):
    """Plot layer-wise cosine similarities (stacked, last layer abs)."""
    print("\nPlotting cosine similarities...")

    n_layers = len(al_stats)
    fig_width = max(16, n_layers * 0.52)
    fig_height = 7.0

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Take absolute value for the last layer
    al_vals = al_stats['mean_value'].values.copy()
    ml_vals = ml_stats['mean_value'].values.copy()
    al_ml_vals = al_plus_ml_stats['mean_value'].values.copy()
    al_vals[-1] = np.abs(al_vals[-1])
    ml_vals[-1] = np.abs(ml_vals[-1])
    al_ml_vals[-1] = np.abs(al_ml_vals[-1])

    x = np.arange(n_layers)
    vals = [al_vals, ml_vals, al_ml_vals]
    colors = [_COLOR_AL, _COLOR_ML, _COLOR_AL_ML]
    labels = [r'$\cos(\mathrm{AL},\ h_l)$',
              r'$\cos(\mathrm{ML},\ h_l)$',
              r'$\cos(\mathrm{AL+ML},\ h_l)$']

    _stacked_bar(ax, x, vals, colors, labels)
    _setup_axes(ax, n_layers, 'Layer', 'Cosine Similarity', labels, colors)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    output_path = os.path.join(PLOT_OUTPUT_DIR, "cosine_similarities.png")
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    plt.savefig(output_path, pad_inches=0.12)
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    """Main execution function."""
    print("=" * 60)
    print("Residual Stream Analysis from Pre-Collected Data")
    print("=" * 60)

    # Check if input files exist
    if not os.path.exists(AL_INPUT) or not os.path.exists(ML_INPUT) or not os.path.exists(AL_PLUS_ML_INPUT):
        print("\nError: Input files not found!")
        print(f"  Expected: {AL_INPUT}")
        print(f"  Expected: {ML_INPUT}")
        print(f"  Expected: {AL_PLUS_ML_INPUT}")
        print("\nPlease run layer_proxy_collection.py first to generate the layer vectors.")
        return

    # Load data and compute
    print("\nLoading pre-collected layer vectors...")

    al_data = load_layer_vectors(AL_INPUT)
    ml_data = load_layer_vectors(ML_INPUT)
    al_plus_ml_data = load_layer_vectors(AL_PLUS_ML_INPUT)

    al_df, ml_df, al_plus_ml_df, al_cos_df, ml_cos_df, al_plus_ml_cos_df = \
        compute_ratios_and_cosine(al_data, ml_data, al_plus_ml_data)

    save_results(al_df, ml_df, al_plus_ml_df, al_cos_df, ml_cos_df, al_plus_ml_cos_df)

    # Aggregate by layer
    al_ratio_stats = aggregate_by_layer(al_df, 'ratio')
    ml_ratio_stats = aggregate_by_layer(ml_df, 'ratio')
    al_plus_ml_ratio_stats = aggregate_by_layer(al_plus_ml_df, 'ratio')

    al_cos_stats = aggregate_by_layer(al_cos_df, 'cosine')
    ml_cos_stats = aggregate_by_layer(ml_cos_df, 'cosine')
    al_plus_ml_cos_stats = aggregate_by_layer(al_plus_ml_cos_df, 'cosine')

    # Print summary
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"\n||AL||²/||h||² - mean: {al_ratio_stats['mean_value'].mean():.4f}")
    print(f"||ML||²/||h||² - mean: {ml_ratio_stats['mean_value'].mean():.4f}")
    print(f"||AL+ML||²/||h||² - mean: {al_plus_ml_ratio_stats['mean_value'].mean():.4f}")
    print(f"\ncos(AL, h_l) - mean: {al_cos_stats['mean_value'].mean():.4f}")
    print(f"cos(ML, h_l) - mean: {ml_cos_stats['mean_value'].mean():.4f}")
    print(f"cos(AL+ML, h_l) - mean: {al_plus_ml_cos_stats['mean_value'].mean():.4f}")

    # Plot
    plot_ratios(al_ratio_stats, ml_ratio_stats, al_plus_ml_ratio_stats)
    plot_cosine(al_cos_stats, ml_cos_stats, al_plus_ml_cos_stats)

    print("\n" + "=" * 60)
    print("All Done!")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - al_ratio.csv, ml_ratio.csv, al_plus_ml_ratio.csv")
    print("  - al_cos.csv, ml_cos.csv, al_plus_ml_cos.csv")
    print("  - residual_stream_ratios.png")
    print("  - cosine_similarities.png")


if __name__ == "__main__":
    main()
