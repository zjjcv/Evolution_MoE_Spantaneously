"""
Plot Syn vs Red scatter plot for MATH dataset (Qwen3 attention heads).

Each point represents one attention head, normalized to [0, 1] range.
Generates scatter plots for each difficulty level (Level 1-5).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plot style
sns.set_theme(style="white")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'sans-serif']

# Configuration
INPUT_BASE_DIR = "/data/zjj/Synergistic_Core/results/MATH"
OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/MATH/plots"

# Level configurations
LEVELS = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]
LEVEL_SUFFIXES = ["Level_1", "Level_2", "Level_3", "Level_4", "Level_5"]


def compute_head_stats_from_pairwise(pairwise_path: str):
    """Compute average syn and red for each head/layer from pairwise data.

    Args:
        pairwise_path: Path to pairwise CSV file

    Returns:
        DataFrame with columns [layer, head, syn, red]
        For layer-level data (ML, AL+ML), head column will be 0 for all rows
    """
    print(f"Loading pairwise data from {pairwise_path}...")
    df = pd.read_csv(pairwise_path)

    print(f"  Total pairs: {len(df):,}")
    print(f"  Questions: {df['question_id'].nunique()}")

    # Check if this is head-level or layer-level data
    if 'head_1' in df.columns:
        # Head-level data (AL): group by (layer_1, head_1)
        head_stats = df.groupby(['layer_1', 'head_1']).agg({
            'syn': 'mean',
            'red': 'mean'
        }).reset_index()
        head_stats.columns = ['layer', 'head', 'syn', 'red']
        print(f"  Unique heads: {len(head_stats)}")
        print(f"  Layers: {head_stats['layer'].nunique()}")
    else:
        # Layer-level data (ML, AL+ML): group by layer_1
        layer_stats = df.groupby(['layer_1']).agg({
            'syn': 'mean',
            'red': 'mean'
        }).reset_index()
        layer_stats.columns = ['layer', 'syn', 'red']
        # Add a dummy head column with value 0
        layer_stats['head'] = 0
        layer_stats = layer_stats[['layer', 'head', 'syn', 'red']]
        print(f"  Unique layers: {len(layer_stats)}")

    return head_stats if 'head_1' in df.columns else layer_stats


def plot_syn_red_scatter(csv_path: str, output_path: str, level_name: str, color_by: str = "ratio"):
    """Plot Syn vs Red scatter plot for attention heads.

    Args:
        csv_path: Path to pairwise CSV file
        output_path: Path to save the plot
        level_name: Name of the difficulty level
        color_by: 'ratio' for syn/(syn+red) ratio rank, 'rank' for (syn-red) rank
    """
    # Compute head statistics from pairwise data
    df = compute_head_stats_from_pairwise(csv_path)

    print(f"Coloring by: {color_by}")

    # Normalize Syn and Red to [0, 1]
    syn_min, syn_max = df['syn'].min(), df['syn'].max()
    red_min, red_max = df['red'].min(), df['red'].max()

    df['syn_norm'] = (df['syn'] - syn_min) / (syn_max - syn_min)
    df['red_norm'] = (df['red'] - red_min) / (red_max - red_min)

    print(f"\nSyn range: [{syn_min:.6f}, {syn_max:.6f}]")
    print(f"Red range: [{red_min:.6f}, {red_max:.6f}]")

    # Calculate rank based on color_by parameter
    if color_by == "ratio":
        # Calculate syn_ratio and rank for coloring
        df['syn_ratio'] = df['syn'] / (df['syn'] + df['red'])
        df['value_rank'] = df['syn_ratio'].rank(method='dense')
        cbar_label = 'Syn/(Syn+Red) Ratio Rank'
    else:
        # Calculate (syn - red) and rank for coloring
        df['syn_minus_red'] = df['syn'] - df['red']
        df['value_rank'] = df['syn_minus_red'].rank(method='dense')
        cbar_label = '(Syn-Red) Rank'

    # Normalize rank to [0, 1] for color mapping
    rank_min, rank_max = df['value_rank'].min(), df['value_rank'].max()
    df['rank_norm'] = (df['value_rank'] - rank_min) / (rank_max - rank_min)

    print(f"Rank range: [{rank_min:.0f}, {rank_max:.0f}]")

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Scatter plot with color based on rank
    # High rank (synergistic) = Deep Red, Low rank (redundant) = Deep Blue
    scatter = ax.scatter(
        df['syn_norm'],
        df['red_norm'],
        c=df['rank_norm'],  # Use normalized rank for color
        cmap='RdBu_r',  # Deep Red (high rank) to Deep Blue (low rank)
        s=30,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.3,
        vmin=0,
        vmax=1
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=12)

    # Add diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='Syn = Red')

    # Add layer depth trajectory (average every 4 layers)
    df['layer_group'] = df['layer'] // 4  # Group layers: 0-3, 4-7, ..., 32-35
    layer_group_stats = df.groupby('layer_group').agg({
        'syn_norm': 'mean',
        'red_norm': 'mean',
        'layer': 'first'  # Get the first layer number in each group
    }).reset_index()
    layer_group_stats = layer_group_stats.sort_values('layer_group')

    # Plot arrows showing depth progression
    for i in range(len(layer_group_stats) - 1):
        current = layer_group_stats.iloc[i]
        next_layer = layer_group_stats.iloc[i + 1]

        # Draw arrow from current to next
        ax.annotate('', xy=(next_layer['syn_norm'], next_layer['red_norm']),
                   xytext=(current['syn_norm'], current['red_norm']),
                   arrowprops=dict(arrowstyle='->', color='blue',
                                  lw=2, alpha=0.6),
                   zorder=8)

    # Mark layer groups with color gradient
    for idx, row in enumerate(layer_group_stats.itertuples()):
        # Color gradient from light blue (shallow) to dark blue (deep)
        blue_intensity = 0.3 + 0.7 * (idx / (len(layer_group_stats) - 1))
        color = plt.cm.Blues(blue_intensity)

        ax.scatter(row.syn_norm, row.red_norm,
                  marker='o', s=150, color=color,
                  edgecolors='black', linewidth=1.5, zorder=10,
                  label=f'Layers {int(row.layer)}-{int(row.layer)+3}')
        # Add layer label
        ax.text(row.syn_norm, row.red_norm, f'  L{int(row.layer)}-{int(row.layer)+3}',
               fontsize=8, fontweight='bold', va='center')

    # Add density-based curve (average of min and max y for each x bin)
    num_bins = 50
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    curve_y = []
    curve_weights = []

    for i in range(num_bins):
        # Find points in this bin
        mask = (df['syn_norm'] >= bin_edges[i]) & (df['syn_norm'] < bin_edges[i + 1])
        points_in_bin = df[mask]

        if len(points_in_bin) > 0:
            # Get min and max y in this bin
            y_min = points_in_bin['red_norm'].min()
            y_max = points_in_bin['red_norm'].max()
            # Average of min and max
            y_avg = (y_min + y_max) / 2
            curve_y.append(y_avg)
            # Weight by number of points (density)
            curve_weights.append(len(points_in_bin))
        else:
            curve_y.append(np.nan)
            curve_weights.append(0)

    curve_y = np.array(curve_y)
    curve_weights = np.array(curve_weights)

    # Only plot bins with data
    valid_mask = ~np.isnan(curve_y)
    if valid_mask.any():
        ax.plot(bin_centers[valid_mask], curve_y[valid_mask],
                color='purple', linewidth=2.5, alpha=0.7,
                label='Density Curve (avg of min/max)', zorder=10)

    # Labels and title
    ax.set_xlabel('Synergy (Syn) - Normalized', fontsize=13)
    ax.set_ylabel('Redundancy (Red) - Normalized', fontsize=13)
    ax.set_title(f'Qwen3-8B-Base: Attention Head Syn-Red Distribution\n({len(df)} heads, colored by {cbar_label})',
                 fontsize=14, pad=15)

    # Grid and limits
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.legend(loc='center', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    import sys

    # Check if processing MMLU data
    if len(sys.argv) > 1 and sys.argv[1] == 'mmlu':
        # MMLU data configuration
        MMLU_INPUT_BASE_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/mmlu/pairwise"
        MMLU_OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/mmlu/plots"
        MMLU_PROXY_TYPES = ["al", "ml", "al_plus_ml"]

        print("=" * 60)
        print("MMLU Syn-Red Scatter Plot Generator")
        print("=" * 60)

        # Create output directories
        scatter_dir = os.path.join(MMLU_OUTPUT_DIR, "scatter")
        os.makedirs(scatter_dir, exist_ok=True)

        # Process each proxy type
        for proxy_type in MMLU_PROXY_TYPES:
            print(f"\n{'='*60}")
            print(f"Processing {proxy_type.upper()}")
            print(f"{'='*60}")

            # Build file path for pairwise data
            input_path = os.path.join(MMLU_INPUT_BASE_DIR, f"{proxy_type}_syn_red_pairwise.csv")

            if not os.path.exists(input_path):
                print(f"Warning: Input file not found: {input_path}")
                print(f"   Skipping...")
                continue

            # Generate scatter plot
            title_prefix = f"MMLU {proxy_type.upper()}"
            output_path = os.path.join(scatter_dir, f"mmlu_{proxy_type}_syn_red_scatter.png")
            plot_syn_red_scatter(input_path, output_path, title_prefix, color_by="ratio")

        print("\n" + "=" * 60)
        print("All scatter plots complete!")
        print("=" * 60)
        print(f"\nOutput directory: {scatter_dir}")
        print("\nGenerated files (3 scatter plots):")
        for proxy_type in MMLU_PROXY_TYPES:
            print(f"  - mmlu_{proxy_type}_syn_red_scatter.png")

    elif len(sys.argv) > 1 and sys.argv[1] == 'arc':
        # ARC data configuration
        ARC_INPUT_BASE_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/ai2arc/pairwise"
        ARC_OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/ai2arc/plots"
        ARC_DIFFICULTIES = ["easy", "challenge"]
        ARC_PROXY_TYPES = ["al", "ml", "al_plus_ml"]

        print("=" * 60)
        print("ARC Syn-Red Scatter Plot Generator")
        print("=" * 60)

        # Create output directories
        scatter_dir = os.path.join(ARC_OUTPUT_DIR, "scatter")
        os.makedirs(scatter_dir, exist_ok=True)

        # Process each difficulty and proxy type
        for difficulty in ARC_DIFFICULTIES:
            print(f"\n{'='*60}")
            print(f"Processing {difficulty.upper()}")
            print(f"{'='*60}")

            difficulty_dir = os.path.join(ARC_INPUT_BASE_DIR, difficulty)
            if not os.path.exists(difficulty_dir):
                print(f"Warning: Directory not found: {difficulty_dir}")
                print(f"   Skipping...")
                continue

            for proxy_type in ARC_PROXY_TYPES:
                print(f"\n--- Proxy Type: {proxy_type} ---")

                # Build file path for pairwise data
                input_path = os.path.join(difficulty_dir, f"{proxy_type}_syn_red_pairwise.csv")

                if not os.path.exists(input_path):
                    print(f"Warning: Input file not found: {input_path}")
                    print(f"   Skipping...")
                    continue

                # Generate scatter plot
                title_prefix = f"ARC-{difficulty.capitalize()} {proxy_type.upper()}"
                output_path = os.path.join(scatter_dir, f"arc_{difficulty}_{proxy_type}_syn_red_scatter.png")
                plot_syn_red_scatter(input_path, output_path, title_prefix, color_by="ratio")

        print("\n" + "=" * 60)
        print("All scatter plots complete!")
        print("=" * 60)
        print(f"\nOutput directory: {scatter_dir}")
        print("\nGenerated files (6 scatter plots):")
        for difficulty in ARC_DIFFICULTIES:
            for proxy_type in ARC_PROXY_TYPES:
                print(f"  - arc_{difficulty}_{proxy_type}_syn_red_scatter.png")

    else:
        # MATH data configuration (original)
        print("=" * 60)
        print("MATH Syn-Red Scatter Plot Generator")
        print("=" * 60)

        # Create output directories
        scatter_dir = os.path.join(OUTPUT_DIR, "scatter")
        os.makedirs(scatter_dir, exist_ok=True)

        # Process each level
        for level, level_suffix in zip(LEVELS, LEVEL_SUFFIXES):
            print(f"\n{'='*60}")
            print(f"Processing {level}")
            print(f"{'='*60}")

            level_dir = os.path.join(INPUT_BASE_DIR, level_suffix)
            if not os.path.exists(level_dir):
                print(f"Warning: Directory not found: {level_dir}")
                print(f"   Skipping...")
                continue

            # Build file path for pairwise data
            input_path = os.path.join(level_dir, f"math_{level_suffix.lower()}_al_pairwise.csv")

            if not os.path.exists(input_path):
                print(f"Warning: Input file not found: {input_path}")
                print(f"   Skipping...")
                continue

            # Generate scatter plot colored by syn/(syn+red) ratio rank
            output_path = os.path.join(scatter_dir, f"math_{level_suffix.lower()}_syn_red_scatter.png")
            plot_syn_red_scatter(input_path, output_path, level, color_by="ratio")

        print("\n" + "=" * 60)
        print("All scatter plots complete!")
        print("=" * 60)
        print(f"\nOutput directory: {scatter_dir}")
        print("\nGenerated files (5 scatter plots):")
        for level_suffix in LEVEL_SUFFIXES:
            level_suffix_lower = level_suffix.lower()
            print(f"  - math_{level_suffix_lower}_syn_red_scatter.png")
