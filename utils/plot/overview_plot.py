"""
Generate overview plot: three signal intensity time series with filled bar charts.

Creates three simulated signal datasets with beautiful filled bar visualization.
Publication-quality styling consistent with other plots in the project.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

# ── Colour palette (consistent with other plots) ──────────────────────────────
_COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#F39B7F', '#91D1C2']  # 5 colors


def generate_signal_data(n_points=50, seed=42):
    """Generate simulated signal intensity data with fluctuating patterns.

    Creates five distinct signal patterns with more variation:
    - Signal 1: Multi-frequency oscillation with amplitude modulation
    - Signal 2: Noisy rising trend with periodic drops
    - Signal 3: Multiple overlapping peaks with noise
    - Signal 4: Decaying oscillation with bursts
    - Signal 5: Step-like pattern with noise

    Args:
        n_points: Number of time points
        seed: Random seed for reproducibility

    Returns:
        t: Time array
        signals: Dictionary with five signal arrays
    """
    np.random.seed(seed)

    # Time array
    t = np.linspace(0, 10, n_points)

    # Signal 1: Multi-frequency oscillation with amplitude modulation (highly fluctuating)
    signal_1 = 0.5 + 0.35 * np.sin(2 * np.pi * 1.2 * t) * np.cos(2 * np.pi * 0.3 * t)
    signal_1 += 0.15 * np.sin(2 * np.pi * 2.5 * t)  # Add higher frequency
    signal_1 += 0.08 * np.random.randn(len(t))  # Add noise
    signal_1 = np.clip(signal_1, 0, 1)  # Clip to [0, 1]

    # Signal 2: Noisy rising trend with periodic drops
    base_trend = 0.2 + 0.5 * (1 - np.exp(-0.4 * t))
    oscillation = 0.12 * np.sin(2 * np.pi * 0.9 * t)
    drops = -0.15 * np.exp(-2 * ((t - 3) ** 2)) - 0.12 * np.exp(-2 * ((t - 7) ** 2))
    signal_2 = base_trend + oscillation + drops
    signal_2 += 0.06 * np.random.randn(len(t))  # Add noise
    signal_2 = np.clip(signal_2, 0, 1)

    # Signal 3: Multiple overlapping peaks with noise
    signal_3 = 0.15 + 0.5 * np.exp(-0.8 * ((t - 2) ** 2))
    signal_3 += 0.4 * np.exp(-0.6 * ((t - 5) ** 2))
    signal_3 += 0.35 * np.exp(-0.8 * ((t - 8) ** 2))
    signal_3 += 0.08 * np.sin(2 * np.pi * 1.5 * t)  # Add ripple
    signal_3 += 0.06 * np.random.randn(len(t))  # Add noise
    signal_3 = np.clip(signal_3, 0, 1)

    # Signal 4: Decaying oscillation with bursts
    signal_4 = 0.4 * np.exp(-0.25 * t) * np.cos(2 * np.pi * 0.8 * t) + 0.35
    signal_4 += 0.12 * np.exp(-3 * ((t - 2) ** 2)) + 0.1 * np.exp(-3 * ((t - 6) ** 2))
    signal_4 += 0.07 * np.random.randn(len(t))
    signal_4 = np.clip(signal_4, 0, 1)

    # Signal 5: Step-like pattern with noise and oscillations
    signal_5 = 0.25 + 0.15 * (1 + np.tanh(2 * (t - 3))) + 0.1 * (1 + np.tanh(2 * (t - 7)))
    signal_5 += 0.08 * np.sin(2 * np.pi * 1.8 * t)
    signal_5 += 0.06 * np.random.randn(len(t))
    signal_5 = np.clip(signal_5, 0, 1)

    signals = {
        'Signal 1': signal_1,
        'Signal 2': signal_2,
        'Signal 3': signal_3,
        'Signal 4': signal_4,
        'Signal 5': signal_5
    }

    return t, signals


def plot_overview_bars(t, signals, output_dir):
    """Plot five signal intensity time series as separate filled bar charts.

    Creates five individual figures, one for each signal.

    Args:
        t: Time array
        signals: Dictionary with five signal arrays
        output_dir: Directory to save the figures
    """
    print("Generating overview plots (separate files)...")

    signal_names = list(signals.keys())
    signal_data = list(signals.values())
    colors = _COLORS
    labels = [r'$\mathrm{Signal\ 1}$', r'$\mathrm{Signal\ 2}$', r'$\mathrm{Signal\ 3}$',
              r'$\mathrm{Signal\ 4}$', r'$\mathrm{Signal\ 5}$']
    filenames = ['signal_1.png', 'signal_2.png', 'signal_3.png', 'signal_4.png', 'signal_5.png']

    bar_width = (t[1] - t[0]) * 1.0  # Full width - no gap between bars

    for i, (signal, color, label, filename) in enumerate(zip(signal_data, colors, labels, filenames)):
        # Create individual figure for each signal (wide and short)
        fig, ax = plt.subplots(1, 1, figsize=(16, 4))

        # Plot filled bars (no gaps between bars)
        ax.bar(t, signal, width=bar_width, color=color, alpha=0.85,
               edgecolor='none', linewidth=0)

        # Axis styling
        ax.set_xlabel(r'$t$', fontweight='bold', fontsize=22)
        ax.set_ylabel('Signal Intensity', fontweight='bold', fontsize=22)
        ax.set_xlim(t[0] - 0.2, t[-1] + 0.2)
        ax.set_ylim(0, 1.0)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.set_xticklabels(['0', '2', '4', '6', '8', '10'])

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

        # Save figure
        output_path = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.10, facecolor='white')
        print(f"  Saved: {output_path}")
        plt.close()


def plot_overview_combined(t, signals, output_path):
    """Plot three signals combined in a single stacked/overlaid view.

    Creates an alternative visualization with all three signals in one plot.

    Args:
        t: Time array
        signals: Dictionary with three signal arrays
        output_path: Path to save the figure
    """
    print("Generating combined overview plot...")

    fig, ax = plt.subplots(1, 1, figsize=(14, 7))

    signal_names = list(signals.keys())
    signal_data = list(signals.values())
    colors = [_COLOR_SIGNAL_1, _COLOR_SIGNAL_2, _COLOR_SIGNAL_3]
    labels = [r'$\mathrm{Signal\ 1}$', r'$\mathrm{Signal\ 2}$', r'$\mathrm{Signal\ 3}$']

    bar_width = (t[1] - t[0]) * 0.25

    # Plot each signal as bars with slight offset
    offsets = [-bar_width, 0, bar_width]

    for signal, color, label, offset in zip(signal_data, colors, labels, offsets):
        ax.bar(t + offset, signal, width=bar_width * 0.9, color=color, alpha=0.85,
               edgecolor='none', label=label)

    # Axis styling
    ax.set_xlabel(r'$t$', fontweight='bold', fontsize=22)
    ax.set_ylabel('Signal Intensity', fontweight='bold', fontsize=22)
    ax.set_xlim(t[0] - 0.3, t[-1] + 0.3)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([0, 2, 4, 6, 8, 10])

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
        text.set_fontsize(18)

    plt.tight_layout()

    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.10, facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    """Main execution function."""
    print("=" * 60)
    print("Overview Plot Generator")
    print("=" * 60)

    # Output directory
    output_dir = "/data/zjj/Synergistic_Core/results/Plots/Qwen3_8_Base/Figure1"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    # Generate signal data
    print("\nGenerating signal data...")
    t, signals = generate_signal_data(n_points=50, seed=42)
    print(f"  Time points: {len(t)}")
    print(f"  Signals: {list(signals.keys())}")

    # Plot: Five separate bar charts (individual files)
    plot_overview_bars(t, signals, output_dir)

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print(f"\nGenerated files:")
    print(f"  - signal_1.png")
    print(f"  - signal_2.png")
    print(f"  - signal_3.png")
    print(f"  - signal_4.png")
    print(f"  - signal_5.png")


if __name__ == "__main__":
    main()
