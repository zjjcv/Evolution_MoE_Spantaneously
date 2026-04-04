import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter

# ── Global style: publication-quality, serif font ──────────────────────
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
_COLOR_LINE = '#c0392b'          # deep red for the profile curve
_COLOR_FILL = '#c0392b'          # same, with alpha
_COLOR_DOT = '#2c3e50'           # dark blue-grey for scatter dots


def compute_head_stats_from_pairwise(pairwise_path: str):
    """Compute average syn and red for each head/layer from pairwise data.

    Args:
        pairwise_path: Path to pairwise CSV file

    Returns:
        DataFrame with columns [layer, head, syn, red]
        For layer-level data (ML, AL+ML), head column will be 0 for all rows
    """
    import pandas as pd
    print(f"Loading pairwise data from {pairwise_path}...")
    df = pd.read_csv(pairwise_path)

    print(f"  Total pairs: {len(df):,}")
    print(f"  Questions: {df['question_id'].nunique()}")

    # Check if this is head-level or layer-level data
    if 'head_1' in df.columns:
        head_stats = df.groupby(['layer_1', 'head_1']).agg({
            'syn': 'mean',
            'red': 'mean'
        }).reset_index()
        head_stats.columns = ['Layer', 'Head', 'Syn', 'Red']
        print(f"  Unique heads: {len(head_stats)}")
        print(f"  Layers: {head_stats['Layer'].nunique()}")
    else:
        layer_stats = df.groupby(['layer_1']).agg({
            'syn': 'mean',
            'red': 'mean'
        }).reset_index()
        layer_stats.columns = ['Layer', 'Syn', 'Red']
        layer_stats['Head'] = 0
        layer_stats = layer_stats[['Layer', 'Head', 'Syn', 'Red']]
        print(f"  Unique layers: {len(layer_stats)}")

    return head_stats if 'head_1' in df.columns else layer_stats


def plot_syn_ratio_rank_gsm8k(csv_path, output_dir=None, metric='syn_ratio_rank', level_name=''):
    """Plot Syn/(Syn+Red) Rank or (Syn-Red) Rank for pairwise data.

    Args:
        csv_path: Path to pairwise CSV file
        output_dir: Directory to save plots
        metric: 'syn_ratio_rank' or 'syn_red_rank'
        level_name: Name of the difficulty level
    """
    # 1. Load data - check if it's pairwise or already aggregated
    df_test = pd.read_csv(csv_path, nrows=10)

    is_pairwise = 'question_id' in df_test.columns and 'layer_1' in df_test.columns

    if is_pairwise:
        df = compute_head_stats_from_pairwise(csv_path)
    else:
        df = pd.read_csv(csv_path)
        if 'syn' in df.columns:
            df['Syn'] = df['syn']
        if 'red' in df.columns:
            df['Red'] = df['red']
        if 'layer' in df.columns:
            df['Layer'] = df['layer']
        if 'head' in df.columns:
            df['Head'] = df['head']

    # 2. Compute syn_ratio = Syn / (Syn + Red)
    df['syn_ratio'] = df['Syn'] / (df['Syn'] + df['Red'])

    # 3. Compute rank of syn_ratio
    df['syn_ratio_rank'] = df['syn_ratio'].rank(method='dense')

    # 4. Compute (Syn-Red) and its rank
    df['syn_red_diff'] = df['Syn'] - df['Red']
    df['syn_red_rank'] = df['syn_red_diff'].rank(method='dense')

    # Select metric
    if metric == 'syn_red_rank':
        metric_col = 'syn_red_rank'
        metric_label = '(Syn-Red) Rank'
    else:
        metric_col = 'syn_ratio_rank'
        metric_label = 'Syn/(Syn+Red) Rank'

    # Head level plot: heatmap + layer profile
    heatmap_data = df.pivot(index='Head', columns='Layer', values=metric_col)

    n_layers = heatmap_data.shape[1]
    n_heads = heatmap_data.shape[0]

    # Adaptive figure size for large grids (Qwen3-8B: 36 layers × 32 heads)
    heatmap_width = max(14, n_layers * 0.48)
    heatmap_height = max(6, n_heads * 0.22)
    fig_width = heatmap_width + 12
    fig_height = max(heatmap_height, 5.5)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height),
                                    gridspec_kw={'width_ratios': [heatmap_width, 12],
                                                 'wspace': 0.08},
                                    constrained_layout=True)

    # ── Heatmap ────────────────────────────────────────────────────────
    cmap = sns.color_palette("RdBu_r", as_cmap=True)
    im = sns.heatmap(heatmap_data, cmap=cmap, ax=ax1,
                     cbar_kws={'label': metric_label, 'shrink': 0.82,
                               'aspect': 30},
                     xticklabels=heatmap_data.columns,
                     yticklabels=heatmap_data.index,
                     linewidths=0, linecolor='none')

    # Style colorbar
    cbar = im.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    cbar.outline.set_linewidth(1.2)

    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Head')

    # Thin x/y ticks for large grids
    if n_layers > 20:
        ax1.set_xticks(ax1.get_xticks()[::2])
        ax1.set_xticklabels(heatmap_data.columns[::2])
    if n_heads > 16:
        ax1.set_yticks(ax1.get_yticks()[::2])
        ax1.set_yticklabels(heatmap_data.index[::2])

    ax1.tick_params(axis='both', length=5)

    # ── Layer profile (smoothed) ───────────────────────────────────────
    layer_avg = df.groupby('Layer')[metric_col].mean().reset_index()

    x_norm = (layer_avg['Layer'] - layer_avg['Layer'].min()) / (layer_avg['Layer'].max() - layer_avg['Layer'].min())
    y_min, y_max = layer_avg[metric_col].min(), layer_avg[metric_col].max()
    y_values = (layer_avg[metric_col].values - y_min) / (y_max - y_min)

    window_size = max(4, len(x_norm) // 5)
    y_ma = np.convolve(y_values, np.ones(window_size) / window_size, mode='same')

    win_len = min(15, len(y_ma) // 2 * 2 + 1)
    if win_len % 2 == 0:
        win_len += 1
    y_smooth = savgol_filter(y_ma, window_length=win_len, polyorder=3)

    x_smooth = np.linspace(x_norm.min(), x_norm.max(), 300)
    spline = make_interp_spline(x_norm, y_smooth, k=3)
    y_final = spline(x_smooth)

    ax2.plot(x_smooth, y_final, color=_COLOR_LINE, lw=3.2, zorder=3, solid_capstyle='round')
    ax2.scatter(x_norm, y_values, s=72, color=_COLOR_LINE, edgecolors='darkred',
                linewidths=1.2, zorder=5)
    ax2.fill_between(x_smooth, y_final, 0, color=_COLOR_FILL, alpha=0.12)
    ax2.axhline(0.5, color='#888888', linestyle='--', linewidth=1.2, alpha=0.7, zorder=1)

    ax2.set_xlabel('Normalized Layer')
    ax2.set_ylabel(f'Normalized {metric_label}')
    ax2.set_xlim(0, 1)
    y_pad = 0.05
    ax2.set_ylim(-y_pad, 1 + y_pad)
    ax2.grid(True, linestyle='-', alpha=0.15, linewidth=0.6, color='#555555')
    ax2.tick_params(axis='both', length=5)

    metric_suffix = 'syn_red_diff' if metric == 'syn_red_rank' else 'syn_ratio_rank'
    if level_name:
        level_suffix = level_name.lower().replace(' ', '_')
        output_path = os.path.join(output_dir, f"{level_suffix}_{metric_suffix}_heatmap.png")
    else:
        output_path = os.path.join(output_dir, f"qwen3_gsm8k_al_{metric_suffix}_profile.png")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()


# ======================================================================
# CLI entry point
# ======================================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'gemma3':
        # Gemma3-4B-Instruct GSM8K heatmap generator
        INPUT_BASE_DIR = "/data/zjj/Synergistic_Core/results/Gemma3-4B-Instruct/data/pairwise"
        OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma3-4B-Instruct/plot"
        PROXY_TYPES = ["al", "ml", "al_plus_ml"]

        print("=" * 60)
        print("Gemma3-4B-Instruct GSM8K Heatmap Generator")
        print("=" * 60)

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for proxy_type in PROXY_TYPES:
            print(f"\n--- Proxy Type: {proxy_type} ---")

            input_path = os.path.join(INPUT_BASE_DIR, f"{proxy_type}_syn_red_pairwise.csv")

            if not os.path.exists(input_path):
                print(f"Warning: Input file not found: {input_path}")
                print(f"   Skipping...")
                continue

            title_prefix = f"Gemma3-GSM8K {proxy_type.upper()}"

            print(f"\nPlotting (Syn-Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_path,
                output_dir=OUTPUT_DIR,
                metric='syn_red_rank',
                level_name=title_prefix
            )

            print(f"\nPlotting Syn/(Syn+Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_path,
                output_dir=OUTPUT_DIR,
                metric='syn_ratio_rank',
                level_name=title_prefix
            )

        print("\n" + "=" * 60)
        print("All plots complete!")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print("\nGenerated files (6 heatmaps):")
        for proxy_type in PROXY_TYPES:
            print(f"  - gemma3_gsm8k_{proxy_type}_syn_red_diff_heatmap.png")
            print(f"  - gemma3_gsm8k_{proxy_type}_syn_ratio_rank_heatmap.png")

    elif len(sys.argv) > 1 and sys.argv[1] == 'math':
        # MATH data configuration
        INPUT_BASE_DIR = "/data/zjj/Synergistic_Core/results/MATH"
        OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/MATH/plots/heatmaps"
        LEVELS = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]
        LEVEL_SUFFIXES = ["Level_1", "Level_2", "Level_3", "Level_4", "Level_5"]

        print("=" * 60)
        print("MATH Heatmap Generator")
        print("=" * 60)

        for level, level_suffix in zip(LEVELS, LEVEL_SUFFIXES):
            print(f"\n{'='*60}")
            print(f"Processing {level}")
            print(f"{'='*60}")

            level_dir = os.path.join(INPUT_BASE_DIR, level_suffix)
            input_path = os.path.join(level_dir, f"math_{level_suffix.lower()}_al_pairwise.csv")

            if not os.path.exists(input_path):
                print(f"Warning: Input file not found: {input_path}")
                print(f"   Skipping...")
                continue

            print(f"\nPlotting (Syn-Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_path,
                output_dir=OUTPUT_DIR,
                metric='syn_red_rank',
                level_name=level
            )

            print(f"\nPlotting Syn/(Syn+Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_path,
                output_dir=OUTPUT_DIR,
                metric='syn_ratio_rank',
                level_name=level
            )

        print("\n" + "=" * 60)
        print("All plots complete!")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print("\nGenerated files (10 heatmaps):")
        for level_suffix in LEVEL_SUFFIXES:
            level_suffix_lower = level_suffix.lower()
            print(f"  - math_{level_suffix_lower}_syn_red_diff_heatmap.png")
            print(f"  - math_{level_suffix_lower}_syn_ratio_rank_heatmap.png")

    elif len(sys.argv) > 1 and sys.argv[1] == 'mmlu':
        # MMLU data configuration
        INPUT_BASE_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/mmlu/pairwise"
        OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/mmlu/plots/heatmaps"
        PROXY_TYPES = ["al", "ml", "al_plus_ml"]

        print("=" * 60)
        print("MMLU Heatmap Generator")
        print("=" * 60)

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for proxy_type in PROXY_TYPES:
            print(f"\n--- Proxy Type: {proxy_type} ---")

            input_path = os.path.join(INPUT_BASE_DIR, f"{proxy_type}_syn_red_pairwise.csv")

            if not os.path.exists(input_path):
                print(f"Warning: Input file not found: {input_path}")
                print(f"   Skipping...")
                continue

            title_prefix = f"MMLU {proxy_type.upper()}"

            print(f"\nPlotting (Syn-Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_path,
                output_dir=OUTPUT_DIR,
                metric='syn_red_rank',
                level_name=title_prefix
            )

            print(f"\nPlotting Syn/(Syn+Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_path,
                output_dir=OUTPUT_DIR,
                metric='syn_ratio_rank',
                level_name=title_prefix
            )

        print("\n" + "=" * 60)
        print("All plots complete!")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print("\nGenerated files (6 heatmaps):")
        for proxy_type in PROXY_TYPES:
            print(f"  - mmlu_{proxy_type}_syn_red_diff_heatmap.png")
            print(f"  - mmlu_{proxy_type}_syn_ratio_rank_heatmap.png")

    elif len(sys.argv) > 1 and sys.argv[1] == 'qwen3_gsm8k':
        # ── Qwen3-8B GSM8K publication-quality heatmaps ───────────────
        INPUT_FILE = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/gsm8k/2048_length/pairwise/al_syn_red_pairwise.csv"
        OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Plots/Qwen3_8_Base"

        print("=" * 60)
        print("Qwen3-8B-GSM8K Publication Heatmap Generator")
        print("=" * 60)
        print(f"\nInput:  {INPUT_FILE}")
        print(f"Output: {OUTPUT_DIR}")

        if not os.path.exists(INPUT_FILE):
            print(f"\nError: Input file not found: {INPUT_FILE}")
            exit(1)

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        print("\nPlotting (Syn-Red) Rank...")
        plot_syn_ratio_rank_gsm8k(
            INPUT_FILE,
            output_dir=OUTPUT_DIR,
            metric='syn_red_rank'
        )

        print("\nPlotting Syn/(Syn+Red) Rank...")
        plot_syn_ratio_rank_gsm8k(
            INPUT_FILE,
            output_dir=OUTPUT_DIR,
            metric='syn_ratio_rank'
        )

        print("\n" + "=" * 60)
        print("All plots complete!")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print("\nGenerated files:")
        print(f"  - qwen3_gsm8k_al_syn_red_diff_profile.png")
        print(f"  - qwen3_gsm8k_al_syn_ratio_rank_profile.png")

    elif len(sys.argv) > 1 and sys.argv[1] == 'arc':
        # ── Qwen3-8B ARC (Easy + Challenge) publication-quality heatmaps ─
        ARC_BASE_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/ai2arc/pairwise"
        OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Plots/Qwen3_8_Base"

        ARC_DATASETS = [
            ("ARC-Easy", "easy"),
            ("ARC-Challenge", "challenge")
        ]

        print("=" * 60)
        print("Qwen3-8B ARC Publication Heatmap Generator")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for dataset_name, dataset_dir in ARC_DATASETS:
            print(f"\n{'='*60}")
            print(f"Processing {dataset_name}")
            print(f"{'='*60}")

            input_file = os.path.join(ARC_BASE_DIR, dataset_dir, "al_syn_red_pairwise.csv")

            if not os.path.exists(input_file):
                print(f"Warning: Input file not found: {input_file}")
                print(f"   Skipping...")
                continue

            print(f"Input: {input_file}")

            # Plot (Syn-Red) Rank
            print(f"\nPlotting (Syn-Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_file,
                output_dir=OUTPUT_DIR,
                metric='syn_red_rank',
                level_name=dataset_name
            )

            # Plot Syn/(Syn+Red) Rank
            print(f"\nPlotting Syn/(Syn+Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_file,
                output_dir=OUTPUT_DIR,
                metric='syn_ratio_rank',
                level_name=dataset_name
            )

        print("\n" + "=" * 60)
        print("All plots complete!")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print("\nGenerated files (4 heatmaps):")
        print(f"  - arc_easy_syn_red_diff_heatmap.png")
        print(f"  - arc_easy_syn_ratio_rank_heatmap.png")
        print(f"  - arc_challenge_syn_red_diff_heatmap.png")
        print(f"  - arc_challenge_syn_ratio_rank_heatmap.png")

    elif len(sys.argv) > 1 and sys.argv[1] == 'gemma3_arc':
        # ── Gemma3-4B-IT ARC (Easy + Challenge) heatmaps ─────────────────
        ARC_BASE_DIR = "/root/data1/zjj/Neurlps2026/Results/Gemma3-4B-IT/data"
        OUTPUT_DIR = "/root/data1/zjj/Neurlps2026/Results/Plots/Gemma3-4B-Instruct"

        ARC_DATASETS = [
            ("ARC-Easy", "easy"),
            ("ARC-Challenge", "challenge")
        ]
        print("=" * 60)
        print("Gemma3-4B-IT ARC Heatmap Generator")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for dataset_name, dataset_dir in ARC_DATASETS:
            print(f"\n{'='*60}")
            print(f"Processing {dataset_name}")
            print(f"{'='*60}")

            input_file = os.path.join(ARC_BASE_DIR, dataset_dir, "al_syn_red_pairwise.csv")

            if not os.path.exists(input_file):
                print(f"Warning: Input file not found: {input_file}")
                print(f"   Skipping...")
                continue

            print(f"Input: {input_file}")

            level_name = f"Gemma3-ARC-{dataset_name}"

            # Plot (Syn-Red) Rank
            print(f"\nPlotting (Syn-Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_file,
                output_dir=OUTPUT_DIR,
                metric='syn_red_rank',
                level_name=level_name
            )

            # Plot Syn/(Syn+Red) Rank
            print(f"\nPlotting Syn/(Syn+Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_file,
                output_dir=OUTPUT_DIR,
                metric='syn_ratio_rank',
                level_name=level_name
            )

        print("\n" + "=" * 60)
        print("All plots complete!")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print("\nGenerated files (4 heatmaps):")
        for dataset_dir in ["easy", "challenge"]:
            print(f"  - gemma3_arc_{dataset_dir}_syn_red_diff_heatmap.png")
            print(f"  - gemma3_arc_{dataset_dir}_syn_ratio_rank_heatmap.png")

    elif len(sys.argv) > 1 and sys.argv[1] == 'gemma3_al':
        # ── Gemma3-4B-Instruct GSM8K AL pairwise data ─────────────────
        INPUT_FILE = "/data/zjj/Synergistic_Core/results/Gemma3-4B-Instruct/data/pairwise/al_syn_red_pairwise.csv"
        OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Plots/Gemma3-4B-Instruct"

        print("=" * 60)
        print("Gemma3-4B-Instruct GSM8K AL Heatmap Generator")
        print("=" * 60)
        print(f"\nInput:  {INPUT_FILE}")
        print(f"Output: {OUTPUT_DIR}")

        if not os.path.exists(INPUT_FILE):
            print(f"\nError: Input file not found: {INPUT_FILE}")
            exit(1)

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        print("\nPlotting (Syn-Red) Rank...")
        plot_syn_ratio_rank_gsm8k(
            INPUT_FILE,
            output_dir=OUTPUT_DIR,
            metric='syn_red_rank',
            level_name='Gemma3-GSM8K-AL'
        )

        print("\nPlotting Syn/(Syn+Red) Rank...")
        plot_syn_ratio_rank_gsm8k(
            INPUT_FILE,
            output_dir=OUTPUT_DIR,
            metric='syn_ratio_rank',
            level_name='Gemma3-GSM8K-AL'
        )

        print("\n" + "=" * 60)
        print("All plots complete!")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print("\nGenerated files:")
        print(f"  - gemma3_gsm8k_al_syn_red_diff_heatmap.png")
        print(f"  - gemma3_gsm8k_al_syn_ratio_rank_heatmap.png")

    elif len(sys.argv) > 1 and sys.argv[1] == 'llama3_gsm8k':
        # ── Llama-3.1-8B-Instruct GSM8K heatmaps ──────────────────────────
        INPUT_BASE_DIR = "/root/data1/zjj/Neurlps2026/Results/Llama-3.1-8B-Instruct/data/pairwise"
        OUTPUT_DIR = "/root/data1/zjj/Neurlps2026/Results/Plots/Llama-3.1-8B-Instruct"
        PROXY_TYPES = ["al", "ml", "al_plus_ml"]

        print("=" * 60)
        print("Llama-3.1-8B-Instruct GSM8K Heatmap Generator")
        print("=" * 60)

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for proxy_type in PROXY_TYPES:
            print(f"\n--- Proxy Type: {proxy_type} ---")

            input_path = os.path.join(INPUT_BASE_DIR, f"{proxy_type}_syn_red_pairwise.csv")

            if not os.path.exists(input_path):
                print(f"Warning: Input file not found: {input_path}")
                print(f"   Skipping...")
                continue

            title_prefix = f"Llama3-GSM8K {proxy_type.upper()}"

            print(f"\nPlotting (Syn-Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_path,
                output_dir=OUTPUT_DIR,
                metric='syn_red_rank',
                level_name=title_prefix
            )

            print(f"\nPlotting Syn/(Syn+Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_path,
                output_dir=OUTPUT_DIR,
                metric='syn_ratio_rank',
                level_name=title_prefix
            )

        print("\n" + "=" * 60)
        print("All plots complete!")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print("\nGenerated files (6 heatmaps):")
        for proxy_type in PROXY_TYPES:
            print(f"  - llama3_gsm8k_{proxy_type}_syn_red_diff_heatmap.png")
            print(f"  - llama3_gsm8k_{proxy_type}_syn_ratio_rank_heatmap.png")

    elif len(sys.argv) > 1 and sys.argv[1] == 'llama3_arc':
        # ── Llama-3.1-8B-Instruct ARC (Easy + Challenge) heatmaps ────────
        ARC_BASE_DIR = "/root/data1/zjj/Neurlps2026/Results/Llama-3.1-8B-Instruct/data"
        OUTPUT_DIR = "/root/data1/zjj/Neurlps2026/Results/Plots/Llama-3.1-8B-Instruct"

        ARC_DATASETS = [
            ("ARC-Easy", "easy"),
            ("ARC-Challenge", "challenge")
        ]
        print("=" * 60)
        print("Llama-3.1-8B-Instruct ARC Heatmap Generator")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for dataset_name, dataset_dir in ARC_DATASETS:
            print(f"\n{'='*60}")
            print(f"Processing {dataset_name}")
            print(f"{'='*60}")

            input_file = os.path.join(ARC_BASE_DIR, dataset_dir, "al_syn_red_pairwise.csv")

            if not os.path.exists(input_file):
                print(f"Warning: Input file not found: {input_file}")
                print(f"   Skipping...")
                continue

            print(f"Input: {input_file}")

            level_name = f"Llama3-ARC-{dataset_name}"

            print(f"\nPlotting (Syn-Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_file,
                output_dir=OUTPUT_DIR,
                metric='syn_red_rank',
                level_name=level_name
            )

            print(f"\nPlotting Syn/(Syn+Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_file,
                output_dir=OUTPUT_DIR,
                metric='syn_ratio_rank',
                level_name=level_name
            )

        print("\n" + "=" * 60)
        print("All plots complete!")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print("\nGenerated files (4 heatmaps):")
        for dataset_dir in ["easy", "challenge"]:
            print(f"  - llama3_arc_{dataset_dir}_syn_red_diff_heatmap.png")
            print(f"  - llama3_arc_{dataset_dir}_syn_ratio_rank_heatmap.png")

    else:
        # GSM8K data configuration (original)
        INPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/gsm8k/2048_length/pairwise"
        OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/gsm8k/plots"

        INPUT_FILE = os.path.join(INPUT_DIR, "pairwise_2_syn_red.csv")

        print("=" * 60)
        print("Qwen3-8B-GSM8K Pairwise Syn-Red Rank Plotting")
        print("=" * 60)
        print(f"\nInput: {INPUT_FILE}")
        print(f"Output: {OUTPUT_DIR}")

        if not os.path.exists(INPUT_FILE):
            print(f"\nError: Input file not found: {INPUT_FILE}")
            exit(1)

        print("\nPlotting (Syn-Red) Rank...")
        plot_syn_ratio_rank_gsm8k(
            INPUT_FILE,
            output_dir=OUTPUT_DIR,
            metric='syn_red_rank'
        )

        print("\nPlotting Syn/(Syn+Red) Rank...")
        plot_syn_ratio_rank_gsm8k(
            INPUT_FILE,
            output_dir=OUTPUT_DIR,
            metric='syn_ratio_rank'
        )

        print("\n" + "=" * 60)
        print("All plots complete!")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print("\nGenerated files:")
        print(f"  - qwen3_gsm8k_al_syn_red_rank_profile.png")
        print(f"  - qwen3_gsm8k_al_syn_ratio_rank_profile.png")
