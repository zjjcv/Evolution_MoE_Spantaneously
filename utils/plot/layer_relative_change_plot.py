"""
Plot layer relative change heatmap from ablation data.

Y-axis: Ablated layer (s)
X-axis: Affected layer (l)
Color: Relative change magnitude (upper-right triangle only, l > s)

Usage:
    python utils/plot/layer_relative_change_plot.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Global style (consistent with synergy_core_syn_ratio_rank.py) ──────
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

# ── Paths ───────────────────────────────────────────────────────────────
INPUT_DIR = "/root/data1/zjj/Neurlps2026/Results/Llama-3.1-8B-Instruct/data/IG_Relative/layer_relative_change"
INPUT_FILE = os.path.join(INPUT_DIR, "layer_relative_change.csv")
OUTPUT_DIR = "/root/data1/zjj/Neurlps2026/Results/Plots/Llama-3.1-8B-Instruct"
OUTPUT_HEATMAP = os.path.join(OUTPUT_DIR, "layer_relative_change.png")
OUTPUT_BARS = os.path.join(OUTPUT_DIR, "layer_relative_change_bars.png")

NUM_LAYERS = 32


def load_and_aggregate_data(input_path: str) -> pd.DataFrame:
    """Load and aggregate relative change data by (s, l) pairs."""
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    print(f"  Total records (raw): {len(df):,}")
    print(f"  Questions: {df['question_id'].nunique()}")

    df_filtered = df[df['ablated_layer_s'] > 0].copy()
    print(f"  Records after removing layer 0 ablation: {len(df_filtered):,}")

    aggregated = df_filtered.groupby(['ablated_layer_s', 'affected_layer_l'])['relative_change'].agg([
        'mean', 'std', 'count'
    ]).reset_index()
    aggregated.columns = ['ablated_layer_s', 'affected_layer_l', 'mean_change', 'std_change', 'count']

    print(f"  Unique (s, l) pairs: {len(aggregated)}")
    print(f"  Mean relative change: {aggregated['mean_change'].mean():.6f}")
    print(f"  Max relative change: {aggregated['mean_change'].max():.6f}")

    return aggregated


def create_heatmap_matrix(df: pd.DataFrame, num_layers: int) -> np.ndarray:
    """Create heatmap matrix: matrix[s, l] = relative change."""
    matrix = np.full((num_layers, num_layers), np.nan)
    for _, row in df.iterrows():
        s = int(row['ablated_layer_s'])
        l = int(row['affected_layer_l'])
        if 0 <= s < num_layers and 0 <= l < num_layers and l > s:
            matrix[s, l] = row['mean_change']
    return matrix


def plot_heatmap(matrix: np.ndarray, output_path: str):
    """Plot and save layer relative change heatmap."""
    print("\nPlotting heatmap...")

    n = matrix.shape[0]

    # Use masked array: only show upper triangle (l > s)
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)   # True for l > s
    mask_full = ~mask                                         # True for l <= s (hidden)

    # Normalize
    valid = matrix[~np.isnan(matrix)]
    data_min, data_max = valid.min(), valid.max()
    matrix_norm = (matrix - data_min) / (data_max - data_min)

    # Fill lower triangle with NaN so seaborn masks it
    matrix_display = np.where(mask, matrix_norm, np.nan)

    print(f"  Data range: [{data_min:.4f}, {data_max:.4f}]")

    fig_width = max(12, n * 0.42)
    fig_height = fig_width * 0.88
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

    cmap = sns.color_palette("coolwarm", as_cmap=True)

    im = sns.heatmap(
        matrix_display,
        cmap=cmap,
        ax=ax,
        cbar_kws={'label': 'Normalized Relative Change',
                  'shrink': 0.78, 'aspect': 28},
        vmin=0,
        vmax=1,
        linewidths=0,
        square=True,
        xticklabels=range(n),
        yticklabels=range(n),
    )

    # Style colorbar
    cbar = im.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    cbar.outline.set_linewidth(1.2)

    ax.set_xlabel('Affected Layer (l)')
    ax.set_ylabel('Ablated Layer (s)')

    # Invert y-axis so layer 0 is at top
    ax.invert_yaxis()

    # Thin ticks for 36 layers
    ax.set_xticks(ax.get_xticks()[::2])
    ax.set_xticklabels(range(0, n, 2))
    ax.set_yticks(ax.get_yticks()[::2])
    ax.set_yticklabels(range(0, n, 2))

    ax.tick_params(axis='both', length=5)

    # Diagonal reference line
    ax.plot([-0.5, n - 0.5], [-0.5, n - 0.5],
            color='white', linestyle='--', linewidth=1.5, alpha=0.6)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"  Saved: {output_path}")
    plt.close()


def plot_aggregated_by_layer(df: pd.DataFrame, num_layers: int, output_dir: str):
    """Plot layer-wise aggregation bar charts."""
    print("\nPlotting layer-wise aggregation...")

    ablated_impact = df.groupby('ablated_layer_s')['mean_change'].mean().reset_index()
    affected_impact = df.groupby('affected_layer_l')['mean_change'].mean().reset_index()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 14),
                                    constrained_layout=True,
                                    gridspec_kw={'hspace': 0.25})

    bar_color_top = '#c0392b'
    bar_color_bot = '#2c3e50'

    # Top: ablated layer impact
    ax1.bar(ablated_impact['ablated_layer_s'], ablated_impact['mean_change'],
            color=bar_color_top, alpha=0.85, edgecolor='none', width=0.8)
    ax1.set_xlabel('Ablated Layer (s)')
    ax1.set_ylabel('Mean Relative Change')
    ax1.set_xlim(-0.5, num_layers - 0.5)
    ax1.grid(True, axis='y', linestyle='-', alpha=0.12, linewidth=0.6, color='#555555')
    ax1.tick_params(axis='both', length=5)
    ax1.set_xticks(range(0, num_layers, 2))

    # Bottom: affected layer impact
    ax2.bar(affected_impact['affected_layer_l'], affected_impact['mean_change'],
            color=bar_color_bot, alpha=0.85, edgecolor='none', width=0.8)
    ax2.set_xlabel('Affected Layer (l)')
    ax2.set_ylabel('Mean Relative Change')
    ax2.set_xlim(-0.5, num_layers - 0.5)
    ax2.grid(True, axis='y', linestyle='-', alpha=0.12, linewidth=0.6, color='#555555')
    ax2.tick_params(axis='both', length=5)
    ax2.set_xticks(range(0, num_layers, 2))

    output_path = os.path.join(output_dir, "layer_relative_change_bars.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    print("=" * 60)
    print("Layer Relative Change Plotting")
    print("=" * 60)
    print(f"\nInput: {INPUT_FILE}")
    print(f"Output: {OUTPUT_DIR}")

    if not os.path.exists(INPUT_FILE):
        print(f"\nError: Input file not found: {INPUT_FILE}")
        return

    aggregated_df = load_and_aggregate_data(INPUT_FILE)
    heatmap_matrix = create_heatmap_matrix(aggregated_df, NUM_LAYERS)

    plot_heatmap(heatmap_matrix, OUTPUT_HEATMAP)
    plot_aggregated_by_layer(aggregated_df, NUM_LAYERS, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("All Done!")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - layer_relative_change.png (main heatmap)")
    print("  - layer_relative_change_bars.png (layer-wise aggregation)")


if __name__ == "__main__":
    main()
