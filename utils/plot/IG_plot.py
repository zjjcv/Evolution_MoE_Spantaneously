"""
Plot Integrated Gradients heatmap.

This script reads IG data and generates a heatmap showing the attribution
of each input token to each layer's representation.

Input: /data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/gsm8k/2048_length/IG/IG.csv

Output: /data/zjj/Synergistic_Core/results/Qwen-3-8B-base/plots/IG_heatmap.png

Usage:
    python utils/plot/IG_plot.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Configuration
INPUT_CSV = "/root/data1/zjj/Neurlps2026/Results/Llama-3.1-8B-Instruct/data/IG_Relative/IG/IG.csv"
OUTPUT_PLOT = "/root/data1/zjj/Neurlps2026/Results/Plots/Llama-3.1-8B-Instruct/IG_heatmap.png"

NUM_LAYERS = 32


def load_ig_data(input_path: str) -> pd.DataFrame:
    """Load IG data from CSV file.

    Args:
        input_path: Path to IG CSV file

    Returns:
        DataFrame with IG data
    """
    print(f"Loading IG data from {input_path}...")

    df = pd.read_csv(input_path)

    print(f"  Total records: {len(df):,}")
    print(f"  Layers: {df['layer'].nunique()}")
    print(f"  Token positions: {df['token_position'].nunique()}")

    return df


def prepare_ig_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare IG data for heatmap visualization.

    Args:
        df: Raw IG DataFrame

    Returns:
        DataFrame with layer, token_position, token, ig_value
    """
    print("\nPreparing IG data...")

    print(f"  Records: {len(df):,}")
    print(f"  IG range: [{df['ig_value'].min():.6f}, {df['ig_value'].max():.6f}]")
    print(f"  Mean IG: {df['ig_value'].mean():.6f}")

    return df


def create_heatmap_matrix(df: pd.DataFrame, num_layers: int) -> tuple:
    """Create heatmap matrix from IG data.

    Args:
        df: Prepared IG DataFrame
        num_layers: Number of layers

    Returns:
        Tuple of (matrix, token_labels)
    """
    print("\nCreating heatmap matrix...")

    # Pivot to create layer x token matrix
    matrix = df.pivot(index='layer', columns='token_position', values='ig_value')

    # Get unique tokens for labels
    token_labels = df.groupby('token_position')['token'].first().values

    # Fill missing values with 0
    matrix = matrix.fillna(0)

    # Ensure we have all layers
    for layer in range(num_layers):
        if layer not in matrix.index:
            matrix.loc[layer] = 0

    # Sort by layer
    matrix = matrix.sort_index()

    print(f"  Matrix shape: {matrix.shape}")
    print(f"  Matrix value range: [{matrix.values.min():.6f}, {matrix.values.max():.6f}]")

    return matrix.values, token_labels


def plot_ig_heatmap(matrix: np.ndarray, token_labels: np.ndarray, output_path: str):
    """Plot IG heatmap.

    Args:
        matrix: 2D array [layers, tokens] with IG values
        token_labels: Token labels for x-axis
        output_path: Path to save the plot
    """
    print("\nPlotting IG heatmap...")

    num_tokens = len(token_labels)

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    # Use absolute values for visualization (show magnitude of attribution)
    matrix_abs = np.abs(matrix)

    # Normalize for better visualization
    matrix_norm = (matrix_abs - matrix_abs.min()) / (matrix_abs.max() - matrix_abs.min() + 1e-10)

    # Create heatmap with Blue -> White -> Red colormap (RdBu_r)
    # X-axis: token positions (0, 1, 2, ..., num_tokens-1)
    # Y-axis: layer indices (0, 1, 2, ..., num_layers-1)
    sns.heatmap(
        matrix_norm,
        cmap='RdBu_r',  # Blue (low) -> White (medium) -> Red (high)
        ax=ax,
        cbar_kws={'label': 'Normalized Attribution Magnitude'},
        xticklabels=range(num_tokens),
        yticklabels=range(matrix.shape[0]),
        vmin=0,
        vmax=1
    )

    ax.set_xlabel('Token Position', fontsize=13)
    ax.set_ylabel('Layer Index', fontsize=13)
    ax.set_title(
        'Integrated Gradients: Token Attribution to Layer Representations\n' +
        'Color: Blue (low attribution) → White (medium) → Red (high attribution)',
        fontsize=14,
        pad=15
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    """Main execution function."""
    print("=" * 60)
    print("Integrated Gradients Heatmap Plotting")
    print("=" * 60)
    print(f"\nInput: {INPUT_CSV}")
    print(f"Output: {OUTPUT_PLOT}")

    # Check input file exists
    if not os.path.exists(INPUT_CSV):
        print(f"\nError: Input file not found: {INPUT_CSV}")
        print("Please run src/IG_collection.py first to generate the data.")
        return

    # Load data
    print("\n" + "=" * 60)
    print("Loading IG data...")
    print("=" * 60)

    ig_df = load_ig_data(INPUT_CSV)

    # Prepare data
    prepared_df = prepare_ig_data(ig_df)

    # Create heatmap matrix
    matrix, token_labels = create_heatmap_matrix(prepared_df, NUM_LAYERS)

    # Plot heatmap
    plot_ig_heatmap(matrix, token_labels, OUTPUT_PLOT)

    print("\n" + "=" * 60)
    print("All Done!")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
