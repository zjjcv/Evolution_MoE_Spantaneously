"""
Compute Energy Distribution using Residual Stream Data

Energy definition:
    E(x) = (||x_l|| / ||x_{l-1}||)₂ × (1 - cos(x, h_l))

where:
    - x ∈ {al, ml, al+ml} (layer outputs)
    - x_l = layer l output, x_{l-1} = previous layer output
    - h_l = previous layer's al+ml output (residual stream input at layer l)
    - cos(x, h_l) = cosine similarity between x and h_l

Input data from residual_stream directory:
    - al.csv, ml.csv, al_plus_ml.csv (layer vectors)
    - al_cos.csv, ml_cos.csv, al_plus_ml_cos.csv (cosine with h_l)

Output: Bar chart showing energy distribution across layers for al, ml, and al+ml

Usage:
    python src/energy_distribution.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Global style (consistent with other plots, publication-quality) ─────────
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

# ── Colour palette (consistent with residual_stream_plot.py) ─────────────────
_COLOR_AL = '#E64B35'           # warm red (Nature-style)
_COLOR_ML = '#4DBBD5'           # teal / cyan
_COLOR_AL_ML = '#00A087'        # deep teal-green

# Configuration
INPUT_DIR = "/root/data1/zjj/Neurlps2026/Results/Llama-3.1-8B-Instruct/data/residual_stream"
OUTPUT_DIR = INPUT_DIR
PLOT_OUTPUT_DIR = "/root/data1/zjj/Neurlps2026/Results/Plots/Llama-3.1-8B-Instruct"

# Input files
AL_CSV = os.path.join(INPUT_DIR, "al.csv")
ML_CSV = os.path.join(INPUT_DIR, "ml.csv")
AL_PLUS_ML_CSV = os.path.join(INPUT_DIR, "al_plus_ml.csv")

AL_COS_CSV = os.path.join(INPUT_DIR, "al_cos.csv")
ML_COS_CSV = os.path.join(INPUT_DIR, "ml_cos.csv")
AL_PLUS_ML_COS_CSV = os.path.join(INPUT_DIR, "al_plus_ml_cos.csv")

OUTPUT_FILE = os.path.join(PLOT_OUTPUT_DIR, "energy_distribution.png")

NUM_LAYERS = 32
HIDDEN_SIZE = 4096


def load_layer_vectors(file_path: str) -> dict:
    """Load layer vectors from CSV file.

    Args:
        file_path: Path to CSV file with columns: question_id, layer, dim_0, dim_1, ...

    Returns:
        Dictionary mapping (question_id, layer) -> vector
    """
    print(f"Loading vectors from {os.path.basename(file_path)}...")

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


def load_cosine_data(file_path: str) -> dict:
    """Load cosine similarity data from CSV file.

    Args:
        file_path: Path to CSV file with columns: question_id, layer, cosine

    Returns:
        Dictionary mapping (question_id, layer) -> cosine value
    """
    print(f"Loading cosine data from {os.path.basename(file_path)}...")

    df = pd.read_csv(file_path)

    data_dict = {}
    for _, row in df.iterrows():
        question_id = int(row['question_id'])
        layer = int(row['layer'])
        cosine = float(row['cosine'])
        data_dict[(question_id, layer)] = cosine

    print(f"  Loaded {len(data_dict)} cosine values")
    return data_dict


def compute_energy_per_layer(vectors: dict, cosine_data: dict, num_layers: int) -> pd.DataFrame:
    """Compute energy for each layer: E = (||x_l|| / ||x_{l-1}||) × (1 - cos(x, h_l))

    Args:
        vectors: Dict[(question_id, layer)] -> vector
        cosine_data: Dict[(question_id, layer)] -> cosine(x, h_l)
        num_layers: Number of layers

    Returns:
        DataFrame with columns: question_id, layer, norm, norm_ratio, cosine, energy
    """
    print("\nComputing energy per layer...")

    # First compute norms for all layers
    norms = {}  # (question_id, layer) -> norm
    for (question_id, layer), vector in vectors.items():
        if layer < num_layers:
            norms[(question_id, layer)] = np.linalg.norm(vector)

    results = []

    # Group by question_id
    questions = sorted(set(qid for qid, _ in norms.keys()))

    for question_id in questions:
        # Compute energy for each layer
        for layer in range(num_layers):
            if (question_id, layer) not in vectors:
                continue

            vector = vectors[(question_id, layer)]
            norm = norms[(question_id, layer)]

            # Get cosine similarity
            cosine = cosine_data.get((question_id, layer), 0.0)

            # Compute norm ratio: ||x_l|| / ||x_{l-1}||
            if layer == 0:
                # For layer 0, use ratio = 1 (no previous layer)
                norm_ratio = 1.0
            elif (question_id, layer - 1) in norms:
                norm_prev = norms[(question_id, layer - 1)]
                if norm_prev < 1e-10:
                    norm_ratio = 1.0
                else:
                    norm_ratio = norm / norm_prev
            else:
                norm_ratio = 1.0

            # Compute energy: (||x_l|| / ||x_{l-1}||) × (1 - cos(x, h_l))
            energy = norm_ratio * (1 - cosine)

            results.append({
                'question_id': question_id,
                'layer': layer,
                'norm': norm,
                'norm_ratio': norm_ratio,
                'cosine': cosine,
                'energy': energy
            })

    df = pd.DataFrame(results)
    print(f"  Computed energy for {len(df)} (question, layer) pairs")

    return df


def aggregate_by_layer(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate energy by layer (average across questions).

    Args:
        df: DataFrame with question_id, layer, energy columns

    Returns:
        DataFrame with layer and mean_energy, std_energy
    """
    print("\nAggregating by layer...")

    layer_stats = df.groupby('layer')['energy'].agg(['mean', 'std', 'count']).reset_index()
    layer_stats.columns = ['layer', 'mean_energy', 'std_energy', 'count']

    return layer_stats


def plot_energy_distribution(al_stats: pd.DataFrame, ml_stats: pd.DataFrame,
                              al_plus_ml_stats: pd.DataFrame, output_path: str):
    """Plot energy distribution bar chart (Nature/Science style publication-quality).

    Args:
        al_stats: Aggregated energy stats for AL
        ml_stats: Aggregated energy stats for ML
        al_plus_ml_stats: Aggregated energy stats for AL+ML
        output_path: Path to save the plot
    """
    print("\nPlotting energy distribution...")

    n_layers = len(al_stats)
    fig_width = max(14, n_layers * 0.45)
    fig_height = 6.8

    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    x = np.arange(n_layers)
    width = 0.24

    # Error bars (standard deviation)
    al_yerr = al_stats['std_energy'].values if 'std_energy' in al_stats.columns else None
    ml_yerr = ml_stats['std_energy'].values if 'std_energy' in ml_stats.columns else None
    al_ml_yerr = al_plus_ml_stats['std_energy'].values if 'std_energy' in al_plus_ml_stats.columns else None

    # Plot bars with elegant styling
    # AL bars
    ax.bar(x - width, al_stats['mean_energy'].values, width, yerr=al_yerr,
           label=r'$\mathrm{AL}$',
           color=_COLOR_AL, alpha=0.88, capsize=2.2, error_kw={'linewidth': 1.0, 'alpha': 0.6})
    # ML bars
    ax.bar(x, ml_stats['mean_energy'].values, width, yerr=ml_yerr,
           label=r'$\mathrm{ML}$',
           color=_COLOR_ML, alpha=0.88, capsize=2.2, error_kw={'linewidth': 1.0, 'alpha': 0.6})
    # AL+ML bars
    ax.bar(x + width, al_plus_ml_stats['mean_energy'].values, width, yerr=al_ml_yerr,
           label=r'$\mathrm{AL+ML}$',
           color=_COLOR_AL_ML, alpha=0.88, capsize=2.2, error_kw={'linewidth': 1.0, 'alpha': 0.6})

    # Axis styling - clean and minimal
    ax.set_xlabel('Layer Index', fontweight='bold', fontsize=22)
    ax.set_ylabel('Energy', fontweight='bold', fontsize=22)
    ax.set_xlim(-0.6, n_layers - 0.4)
    ax.set_xticks(range(0, n_layers, 2))
    ax.set_xticklabels(range(0, n_layers, 2))

    # Set y-axis ticks at 0.5 intervals
    y_max = ax.get_ylim()[1]
    y_ticks = np.arange(0, y_max + 0.5, 0.5)
    ax.set_yticks(y_ticks)

    # Subtle grid
    ax.grid(True, axis='y', linestyle='-', alpha=0.08, linewidth=0.8, color='#888888')
    ax.set_axisbelow(True)

    # Tick styling
    ax.tick_params(axis='both', which='major', length=4.5, width=1.2, direction='in')
    ax.tick_params(axis='both', which='minor', length=0)

    # Spine styling - remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.3)
    ax.spines['bottom'].set_linewidth(1.3)

    # Elegant legend - placed outside plot area
    leg = ax.legend(
        frameon=True, fancybox=False, edgecolor='#888888',
        framealpha=0.96, loc='upper center', ncol=3,
        bbox_to_anchor=(0.5, 1.0),
        handletextpad=0.6, columnspacing=2.0, borderpad=0.5,
        handlelength=1.4, handleheight=1.0,
    )
    leg.get_frame().set_linewidth(1.0)
    for text in leg.get_texts():
        text.set_fontweight('bold')
        text.set_fontsize(20)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.10, facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    """Main execution function."""
    print("=" * 60)
    print("Energy Distribution Analysis from Residual Stream Data")
    print("=" * 60)
    print(f"\nInput directory: {INPUT_DIR}")
    print(f"Output file: {OUTPUT_FILE}")

    # Check input files exist
    required_files = [AL_CSV, ML_CSV, AL_PLUS_ML_CSV, AL_COS_CSV, ML_COS_CSV, AL_PLUS_ML_COS_CSV]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print("\nError: Required input files not found:")
        for f in missing_files:
            print(f"  {f}")
        print("\nPlease run residual_stream_plot.py first to generate the required data.")
        return

    # Load vectors
    print("\n" + "=" * 60)
    print("Loading layer vectors...")
    print("=" * 60)

    al_vectors = load_layer_vectors(AL_CSV)
    ml_vectors = load_layer_vectors(ML_CSV)
    al_plus_ml_vectors = load_layer_vectors(AL_PLUS_ML_CSV)

    # Load cosine data
    print("\n" + "=" * 60)
    print("Loading cosine similarity data...")
    print("=" * 60)

    al_cosine = load_cosine_data(AL_COS_CSV)
    ml_cosine = load_cosine_data(ML_COS_CSV)
    al_plus_ml_cosine = load_cosine_data(AL_PLUS_ML_COS_CSV)

    # Compute energy for each type
    print("\n" + "=" * 60)
    print("Computing energy distribution...")
    print("=" * 60)

    al_energy_df = compute_energy_per_layer(al_vectors, al_cosine, NUM_LAYERS)
    ml_energy_df = compute_energy_per_layer(ml_vectors, ml_cosine, NUM_LAYERS)
    al_plus_ml_energy_df = compute_energy_per_layer(al_plus_ml_vectors, al_plus_ml_cosine, NUM_LAYERS)

    # Aggregate by layer
    al_stats = aggregate_by_layer(al_energy_df)
    ml_stats = aggregate_by_layer(ml_energy_df)
    al_plus_ml_stats = aggregate_by_layer(al_plus_ml_energy_df)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"\nAL Energy:")
    print(f"  Mean: {al_stats['mean_energy'].mean():.4f}")
    print(f"  Max (layer {al_stats.loc[al_stats['mean_energy'].idxmax(), 'layer']}): {al_stats['mean_energy'].max():.4f}")
    print(f"  Min (layer {al_stats.loc[al_stats['mean_energy'].idxmin(), 'layer']}): {al_stats['mean_energy'].min():.4f}")

    print(f"\nML Energy:")
    print(f"  Mean: {ml_stats['mean_energy'].mean():.4f}")
    print(f"  Max (layer {ml_stats.loc[ml_stats['mean_energy'].idxmax(), 'layer']}): {ml_stats['mean_energy'].max():.4f}")
    print(f"  Min (layer {ml_stats.loc[ml_stats['mean_energy'].idxmin(), 'layer']}): {ml_stats['mean_energy'].min():.4f}")

    print(f"\nAL+ML Energy:")
    print(f"  Mean: {al_plus_ml_stats['mean_energy'].mean():.4f}")
    print(f"  Max (layer {al_plus_ml_stats.loc[al_plus_ml_stats['mean_energy'].idxmax(), 'layer']}): {al_plus_ml_stats['mean_energy'].max():.4f}")
    print(f"  Min (layer {al_plus_ml_stats.loc[al_plus_ml_stats['mean_energy'].idxmin(), 'layer']}): {al_plus_ml_stats['mean_energy'].min():.4f}")

    # Plot
    plot_energy_distribution(al_stats, ml_stats, al_plus_ml_stats, OUTPUT_FILE)

    print("\n" + "=" * 60)
    print("All Done!")
    print("=" * 60)
    print(f"\nOutput saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
