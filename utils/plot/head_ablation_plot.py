"""
Plot head ablation results with publication-quality styling.

Visualizes the impact of different ablation strategies on model accuracy.
Nature/Science style visualization.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

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

# ── Colour palette ──────────────────────────────────────────────────────────
_COLOR_H2L = '#E64B35'         # warm red for high-to-low
_COLOR_L2H = '#4DBBD5'         # teal for low-to-high
_COLOR_RANDOM = '#00A087'      # deep teal-green for random
_COLOR_BASELINE = '#888888'    # gray for baseline


# Configuration (can be overridden by command line)
DEFAULT_INPUT_CSV = "/root/data1/zjj/Neurlps2026/Results/Llama-3.1-8B-Instruct/ablation/ablation_results.csv"
DEFAULT_OUTPUT_DIR = "/root/data1/zjj/Neurlps2026/Results/Plots/Llama-3.1-8B-Instruct"


def load_ablation_data(csv_path: str) -> pd.DataFrame:
    """Load ablation results from CSV."""
    print(f"Loading data from {csv_path}...")

    df = pd.read_csv(csv_path)

    print(f"  Total records: {len(df)}")
    print(f"  Experiment types: {df['experiment_type'].unique()}")
    print(f"  Ablation range: {df['pct_ablated'].min():.1f}% - {df['pct_ablated'].max():.1f}%")

    return df


def prepare_plot_data(df: pd.DataFrame) -> dict:
    """Prepare data for plotting by aggregating random experiments."""
    print("\nPreparing plot data...")

    plot_data = {}

    # High-to-Low Syn-Red Rank
    h2l_df = df[df['experiment_type'] == 'high_to_low'].copy()
    if len(h2l_df) > 0:
        plot_data['high_to_low'] = {
            'x': h2l_df['pct_ablated'].values,
            'y': h2l_df['accuracy'].values,
            'label': r'$\mathrm{High\!-\!to\!-\!Low\ (Synergistic\ first)}$',
            'color': _COLOR_H2L
        }

    # Low-to-High Syn-Red Rank
    l2h_df = df[df['experiment_type'] == 'low_to_high'].copy()
    if len(l2h_df) > 0:
        plot_data['low_to_high'] = {
            'x': l2h_df['pct_ablated'].values,
            'y': l2h_df['accuracy'].values,
            'label': r'$\mathrm{Low\!-\!to\!-\!High\ (Redundant\ first)}$',
            'color': _COLOR_L2H
        }

    # Random ablation (aggregate over repeats)
    random_df = df[df['experiment_type'] == 'random'].copy()
    if len(random_df) > 0:
        random_agg = random_df.groupby('pct_ablated')['accuracy'].agg(['mean', 'std', 'count']).reset_index()
        plot_data['random'] = {
            'x': random_agg['pct_ablated'].values,
            'y_mean': random_agg['mean'].values,
            'y_std': random_agg['std'].values,
            'label': r'$\mathrm{Random}$',
            'color': _COLOR_RANDOM
        }

    # Baseline
    if 'baseline_acc' in df.columns or len(df) > 0:
        baseline_df = df[df['pct_ablated'] < 0.5]
        if len(baseline_df) > 0:
            baseline_acc = baseline_df['accuracy'].values[0]
            plot_data['baseline'] = baseline_acc
            print(f"  Baseline accuracy: {baseline_acc:.4f}")

    return plot_data


def plot_ablation_curves_publication(plot_data: dict, output_path: str, dataset_name: str = "Qwen3-8B"):
    """Plot ablation curves with Nature/Science style."""
    print(f"\nPlotting ablation curves (publication style)...")

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    # Plot baseline
    if 'baseline' in plot_data:
        baseline = plot_data['baseline']
        ax.axhline(y=baseline, color=_COLOR_BASELINE, linestyle='--', alpha=0.6,
                  linewidth=1.8, label=rf'$\mathrm{{Baseline\ ({baseline:.3f})}}$')

    # Plot High-to-Low
    if 'high_to_low' in plot_data:
        data = plot_data['high_to_low']
        ax.plot(data['x'], data['y'], color=data['color'],
                linewidth=2.8, marker='o', markersize=6, markeredgewidth=1.5,
                markeredgecolor='white', label=data['label'], zorder=5)

    # Plot Low-to-High
    if 'low_to_high' in plot_data:
        data = plot_data['low_to_high']
        ax.plot(data['x'], data['y'], color=data['color'],
                linewidth=2.8, marker='s', markersize=6, markeredgewidth=1.5,
                markeredgecolor='white', label=data['label'], zorder=4)

    # Plot Random with error band
    if 'random' in plot_data:
        data = plot_data['random']
        ax.plot(data['x'], data['y_mean'], color=data['color'],
                linewidth=2.8, marker='^', markersize=6, markeredgewidth=1.5,
                markeredgecolor='white', label=data['label'], zorder=3)
        # Fill standard deviation
        ax.fill_between(data['x'],
                       data['y_mean'] - data['y_std'],
                       data['y_mean'] + data['y_std'],
                       color=data['color'], alpha=0.18, linewidth=0, zorder=2)

    # Axis styling - clean and minimal
    ax.set_xlabel('Percentage of Heads Ablated (%)', fontweight='bold', fontsize=22)
    ax.set_ylabel('Accuracy', fontweight='bold', fontsize=22)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)

    # Subtle grid
    ax.grid(True, axis='y', linestyle='-', alpha=0.08, linewidth=0.8, color='#888888')
    ax.set_axisbelow(True)

    # Tick styling
    ax.tick_params(axis='both', which='major', length=4.5, width=1.2, direction='in')

    # Spine styling - remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.3)
    ax.spines['bottom'].set_linewidth(1.3)

    # Legend
    leg = ax.legend(
        frameon=True, fancybox=False, edgecolor='#888888',
        framealpha=0.96, loc='upper right',
        handletextpad=0.6, borderpad=0.5,
        handlelength=1.4, handleheight=1.0,
    )
    leg.get_frame().set_linewidth(1.0)
    for text in leg.get_texts():
        text.set_fontweight('bold')
        text.set_fontsize(16)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.10, facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_accuracy_drop_publication(plot_data: dict, output_path: str):
    """Plot accuracy drop from baseline with Nature/Science style."""
    print(f"\nPlotting accuracy drop (publication style)...")

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    baseline = plot_data.get('baseline', 0)

    # Plot High-to-Low
    if 'high_to_low' in plot_data:
        data = plot_data['high_to_low']
        drop = baseline - data['y']
        ax.plot(data['x'], drop, color=data['color'],
                linewidth=2.8, marker='o', markersize=6, markeredgewidth=1.5,
                markeredgecolor='white', label=r'$\mathrm{High\!-\!to\!-\!Low}$', zorder=5)

    # Plot Low-to-High
    if 'low_to_high' in plot_data:
        data = plot_data['low_to_high']
        drop = baseline - data['y']
        ax.plot(data['x'], drop, color=data['color'],
                linewidth=2.8, marker='s', markersize=6, markeredgewidth=1.5,
                markeredgecolor='white', label=r'$\mathrm{Low\!-\!to\!-\!High}$', zorder=4)

    # Plot Random with error band
    if 'random' in plot_data:
        data = plot_data['random']
        drop_mean = baseline - data['y_mean']
        ax.plot(data['x'], drop_mean, color=data['color'],
                linewidth=2.8, marker='^', markersize=6, markeredgewidth=1.5,
                markeredgecolor='white', label=r'$\mathrm{Random}$', zorder=3)
        ax.fill_between(data['x'],
                       drop_mean - data['y_std'],
                       drop_mean + data['y_std'],
                       color=data['color'], alpha=0.18, linewidth=0, zorder=2)

    # Axis styling
    ax.set_xlabel('Percentage of Heads Ablated (%)', fontweight='bold', fontsize=22)
    ax.set_ylabel('Accuracy Drop from Baseline', fontweight='bold', fontsize=22)
    ax.set_xlim(0, 100)

    # Subtle grid
    ax.grid(True, axis='y', linestyle='-', alpha=0.08, linewidth=0.8, color='#888888')
    ax.set_axisbelow(True)

    # Tick styling
    ax.tick_params(axis='both', which='major', length=4.5, width=1.2, direction='in')

    # Spine styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.3)
    ax.spines['bottom'].set_linewidth(1.3)

    # Legend
    leg = ax.legend(
        frameon=True, fancybox=False, edgecolor='#888888',
        framealpha=0.96, loc='upper left',
        handletextpad=0.6, borderpad=0.5,
        handlelength=1.4, handleheight=1.0,
    )
    leg.get_frame().set_linewidth(1.0)
    for text in leg.get_texts():
        text.set_fontweight('bold')
        text.set_fontsize(16)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.10, facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_key_levels_comparison_publication(plot_data: dict, output_path: str):
    """Plot bar chart comparing accuracy at key ablation levels (Nature/Science style)."""
    print(f"\nPlotting key levels comparison (publication style)...")

    key_levels = [10, 20, 30, 40, 50, 70, 90]

    # Prepare data
    comparison_data = []

    for level in key_levels:
        for exp_type, exp_name in [
            ('high_to_low', 'H2L'),
            ('low_to_high', 'L2H'),
            ('random', 'Random')
        ]:
            if exp_type not in plot_data:
                continue

            data = plot_data[exp_type]
            idx = (np.abs(data['x'] - level)).argmin()

            if exp_type == 'random':
                acc = data['y_mean'][idx]
                std = data['y_std'][idx]
                comparison_data.append({
                    'Level': f'{level}%',
                    'Strategy': exp_name,
                    'Accuracy': acc,
                    'Std': std
                })
            else:
                acc = data['y'][idx]
                comparison_data.append({
                    'Level': f'{level}%',
                    'Strategy': exp_name,
                    'Accuracy': acc,
                    'Std': 0
                })

    df_comp = pd.DataFrame(comparison_data)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(13, 7))

    # Pivot for plotting
    pivot_df = df_comp.pivot(index='Level', columns='Strategy', values='Accuracy')

    # Define colors
    colors = [_COLOR_H2L, _COLOR_L2H, _COLOR_RANDOM]

    # Plot bars
    pivot_df.plot(kind='bar', ax=ax, color=colors, alpha=0.88,
                 edgecolor='none', width=0.7)

    # Add baseline line if available
    if 'baseline' in plot_data:
        ax.axhline(y=plot_data['baseline'], color=_COLOR_BASELINE,
                  linestyle='--', alpha=0.6, linewidth=1.8, label='Baseline')

    # Axis styling
    ax.set_xlabel('Heads Ablated (%)', fontweight='bold', fontsize=22)
    ax.set_ylabel('Accuracy', fontweight='bold', fontsize=22)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')

    # Subtle grid
    ax.grid(True, axis='y', linestyle='-', alpha=0.08, linewidth=0.8, color='#888888')
    ax.set_axisbelow(True)

    # Tick styling
    ax.tick_params(axis='both', which='major', length=4.5, width=1.2, direction='in')

    # Spine styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.3)
    ax.spines['bottom'].set_linewidth(1.3)

    # Legend
    leg = ax.legend(
        frameon=True, fancybox=False, edgecolor='#888888',
        framealpha=0.96, loc='upper right',
        handletextpad=0.6, borderpad=0.5,
        handlelength=1.4, handleheight=1.0,
    )
    leg.get_frame().set_linewidth(1.0)
    for text in leg.get_texts():
        text.set_fontweight('bold')
        text.set_fontsize(16)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.10, facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()


def generate_summary_report(df: pd.DataFrame, plot_data: dict, output_path: str):
    """Generate a text summary report."""
    print(f"\nGenerating summary report...")

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("HEAD ABLATION EXPERIMENT SUMMARY")
    report_lines.append("=" * 70)
    report_lines.append("")

    # Baseline
    if 'baseline' in plot_data:
        report_lines.append(f"Baseline Accuracy: {plot_data['baseline']:.4f}")
        report_lines.append("")

    # Comparison at key levels
    report_lines.append("Accuracy at Key Ablation Levels:")
    report_lines.append("-" * 70)

    key_levels = [10, 20, 30, 50, 70, 90]

    for level in key_levels:
        report_lines.append(f"\n{level}% Ablation:")

        for exp_type, exp_name in [
            ('high_to_low', 'High-to-Low Syn-Red'),
            ('low_to_high', 'Low-to-High Syn-Red'),
            ('random', 'Random')
        ]:
            if exp_type not in plot_data:
                continue

            data = plot_data[exp_type]

            if exp_type == 'random':
                idx = (np.abs(data['x'] - level)).argmin()
                acc = data['y_mean'][idx]
                std = data['y_std'][idx]
                report_lines.append(f"  {exp_name:25s}: {acc:.4f} ± {std:.4f}")
            else:
                idx = (np.abs(data['x'] - level)).argmin()
                acc = data['y'][idx]
                drop = plot_data.get('baseline', 0) - acc
                report_lines.append(f"  {exp_name:25s}: {acc:.4f} (drop: {drop:.4f})")

    report_lines.append("")
    report_lines.append("=" * 70)

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"  Saved: {output_path}")

    # Print to console
    print("\n" + '\n'.join(report_lines))


def main():
    """Main execution function."""
    # Parse command line arguments
    input_csv = DEFAULT_INPUT_CSV
    output_dir = DEFAULT_OUTPUT_DIR

    if len(sys.argv) > 1:
        input_csv = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    print("=" * 70)
    print("Head Ablation Results Visualization (Publication-Quality)")
    print("=" * 70)
    print(f"\nInput: {input_csv}")
    print(f"Output directory: {output_dir}")

    # Check input file exists
    if not os.path.exists(input_csv):
        print(f"\nError: Input file not found: {input_csv}")
        print("Please run src/simple_layer_ablation.py first to generate the data.")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = load_ablation_data(input_csv)

    # Prepare plot data
    plot_data = prepare_plot_data(df)

    # Generate output paths
    output_curve = os.path.join(output_dir, "head_ablation_curve.png")
    output_drop = os.path.join(output_dir, "head_ablation_curve_drop.png")
    output_bar = os.path.join(output_dir, "head_ablation_bar_comparison.png")
    report_path = os.path.join(output_dir, "ablation_summary.txt")

    # Plot 1: Main ablation curves
    plot_ablation_curves_publication(plot_data, output_curve)

    # Plot 2: Accuracy drop
    plot_accuracy_drop_publication(plot_data, output_drop)

    # Plot 3: Bar comparison at key levels
    plot_key_levels_comparison_publication(plot_data, output_bar)

    # Generate summary report
    generate_summary_report(df, plot_data, report_path)

    print("\n" + "=" * 70)
    print("All Done!")
    print("=" * 70)
    print(f"\nGenerated files:")
    print(f"  - head_ablation_curve.png (main accuracy curve)")
    print(f"  - head_ablation_curve_drop.png (accuracy drop from baseline)")
    print(f"  - head_ablation_bar_comparison.png (bar chart at key levels)")
    print(f"  - ablation_summary.txt (text summary)")


if __name__ == "__main__":
    main()
