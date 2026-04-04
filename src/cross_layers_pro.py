"""
Compute Cross-layer Projection (Alignment) for GSM8K proxy data.

This script reads 3 CSV files (al, ml, al+ml for GSM8K),
computes the alignment (cosine similarity) between each pair of vectors:
- al: head-to-head alignment
- ml/al+ml: layer-to-layer alignment

Formula: Align(U_L, U_{L+1}) = (V_L · V_{L+1}) / (||V_L|| ||V_{L+1}||)

Output: 3 CSV files with alignment matrices and 3 heatmap figures.

Usage:
    python src/cross_layers_pro.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================
# Configuration
# ============================================================
INPUT_DIR = "/root/data1/zjj/Neurlps2026/Results/Llama-3.1-8B-Instruct/data/residual_stream"
OUTPUT_DIR = "/root/data1/zjj/Neurlps2026/Results/Llama-3.1-8B-Instruct/data/residual_stream"
FIGURE_DIR = "/root/data1/zjj/Neurlps2026/Results/Plots/Llama-3.1-8B-Instruct"

# File configurations
FILES_CONFIG = [
    {"name": "gsm8k_al", "type": "al", "per_head": True},
    {"name": "gsm8k_ml", "type": "ml", "per_head": False},
    {"name": "gsm8k_al_plus_ml", "type": "al_plus_ml", "per_head": False},
]


def load_csv_data(csv_path: str, per_head: bool) -> pd.DataFrame:
    """Load CSV data and extract vector information.

    Args:
        csv_path: Path to CSV file
        per_head: If True, data is per-head (has 'head' column)

    Returns:
        DataFrame with component identifiers and vectors
    """
    df = pd.read_csv(csv_path)

    # Get dimension columns
    dim_cols = [col for col in df.columns if col.startswith('dim_')]

    # Create component identifier
    if per_head:
        df['component_uid'] = df['layer'].astype(str) + '_' + df['head'].astype(str)
    else:
        df['component_uid'] = df['layer'].astype(str)

    return df


def extract_vectors(df: pd.DataFrame, per_head: bool) -> dict:
    """Extract vectors from DataFrame.

    Args:
        df: DataFrame with vector data
        per_head: If True, data is per-head

    Returns:
        Dictionary mapping component_id to vector array
    """
    dim_cols = [col for col in df.columns if col.startswith('dim_')]
    vectors = {}

    for _, row in df.iterrows():
        comp_id = row['component_uid']
        vector = row[dim_cols].values.astype(np.float64)
        vectors[comp_id] = vector

    return vectors


def compute_alignment_matrix(vectors: dict, per_head: bool) -> pd.DataFrame:
    """Compute alignment (cosine similarity) matrix between all pairs.

    Args:
        vectors: Dictionary mapping component_id to vector
        per_head: If True, components are heads

    Returns:
        DataFrame with alignment scores
    """
    # Get sorted component IDs
    comp_ids = sorted(vectors.keys())

    # Initialize matrix
    n = len(comp_ids)
    alignment_matrix = np.zeros((n, n))

    # Compute alignment for each pair
    for i, id1 in enumerate(comp_ids):
        vec1 = vectors[id1]
        norm1 = np.linalg.norm(vec1)

        for j, id2 in enumerate(comp_ids):
            vec2 = vectors[id2]
            norm2 = np.linalg.norm(vec2)

            # Cosine similarity: (v1 · v2) / (||v1|| ||v2||)
            if norm1 > 0 and norm2 > 0:
                alignment = np.dot(vec1, vec2) / (norm1 * norm2)
            else:
                alignment = 0.0

            alignment_matrix[i, j] = alignment

    # Create DataFrame
    df = pd.DataFrame(alignment_matrix, index=comp_ids, columns=comp_ids)

    return df


def parse_component_id(comp_id: str, per_head: bool) -> tuple:
    """Parse component ID to extract layer and head info.

    Args:
        comp_id: Component identifier (e.g., "0" or "0_0")
        per_head: If True, expect head in ID

    Returns:
        (layer, head) or (layer,) tuple
    """
    if per_head:
        layer, head = map(int, comp_id.split('_'))
        return layer, head
    else:
        return int(comp_id),


def save_alignment_csv(alignment_df: pd.DataFrame, output_path: str) -> None:
    """Save alignment matrix to CSV.

    Args:
        alignment_df: Alignment matrix DataFrame
        output_path: Path to save CSV
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    alignment_df.to_csv(output_path)
    print(f"Saved: {output_path}")


def plot_alignment_heatmap(alignment_df: pd.DataFrame, title: str,
                           per_head: bool, output_path: str,
                           vmin: float = -1, vmax: float = 1) -> None:
    """Plot alignment heatmap and save to file.

    Args:
        alignment_df: Alignment matrix DataFrame
        title: Plot title
        per_head: If True, components are heads
        output_path: Path to save figure
        vmin: Minimum value for colorbar
        vmax: Maximum value for colorbar
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create labels
    labels = alignment_df.index.tolist()

    # For per-head, use simpler labels to avoid clutter
    if per_head and len(labels) > 50:
        # Show only every nth label
        step = max(1, len(labels) // 20)
        labels = [l if i % step == 0 else '' for i, l in enumerate(labels)]

    # Plot heatmap
    sns.heatmap(alignment_df.values,
                ax=ax,
                cmap='RdBu_r',
                vmin=vmin,
                vmax=vmax,
                xticklabels=False,  # Hide x labels for clarity
                yticklabels=False,
                cbar=True,
                cbar_kws={'label': 'Alignment (Cosine Similarity)'})

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Components', fontsize=12)
    ax.set_ylabel('Components', fontsize=12)

    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {output_path}")

    # Also save as PDF
    output_pdf = output_path.replace('.png', '.pdf')
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"Saved figure: {output_pdf}")

    plt.close(fig)


def main():
    """Main execution function."""
    print("=" * 60)
    print("Cross-layer Projection (Alignment) Computation")
    print("=" * 60)

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    # Process each file
    for config in FILES_CONFIG:
        file_name = config['name']
        file_type = config['type']
        per_head = config['per_head']

        print(f"\nProcessing: {file_name}")

        # Build file path
        input_path = os.path.join(INPUT_DIR, f"{file_name}.csv")

        if not os.path.exists(input_path):
            print(f"  Warning: File not found: {input_path}")
            continue

        # Load data
        df = load_csv_data(input_path, per_head)
        print(f"  Loaded: {len(df)} rows (questions x components)")

        # Get unique question IDs
        if 'question_id' in df.columns:
            question_ids = df['question_id'].unique()
            print(f"  Found: {len(question_ids)} questions")
        else:
            print(f"  Warning: No question_id column found, treating all data as single question")
            question_ids = [None]

        # Compute alignment matrix for each question, then average
        alignment_matrices = []

        for q_id in question_ids:
            if q_id is not None:
                # Filter data for this question
                q_df = df[df['question_id'] == q_id].copy()
            else:
                q_df = df.copy()

            # Extract vectors
            vectors = extract_vectors(q_df, per_head)

            # Compute alignment matrix
            alignment_df = compute_alignment_matrix(vectors, per_head)
            alignment_matrices.append(alignment_df.values)

        # Average alignment matrices across questions
        avg_alignment_matrix = np.mean(alignment_matrices, axis=0)

        # Create DataFrame with average alignment
        # Get component IDs from first question
        if question_ids[0] is not None:
            first_q_df = df[df['question_id'] == question_ids[0]].copy()
        else:
            first_q_df = df.copy()

        first_vectors = extract_vectors(first_q_df, per_head)
        comp_ids = sorted(first_vectors.keys())

        avg_alignment_df = pd.DataFrame(avg_alignment_matrix, index=comp_ids, columns=comp_ids)
        print(f"  Average alignment matrix shape: {avg_alignment_df.shape}")

        # Save averaged alignment to CSV (overwrite original file)
        save_alignment_csv(avg_alignment_df, input_path)

        # Also save to output directory for backup
        output_path = os.path.join(OUTPUT_DIR, f"{file_name}_alignment.csv")
        save_alignment_csv(avg_alignment_df, output_path)

        # Create title
        comp_type = file_type.replace('_', ' ').upper()
        title = f"GSM8K - {comp_type}"

        # Plot and save heatmap
        figure_path = os.path.join(FIGURE_DIR, f"{file_name}_alignment.png")
        plot_alignment_heatmap(avg_alignment_df, title, per_head, figure_path)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"\nAlignment directory: {OUTPUT_DIR}")
    print(f"Figure directory: {FIGURE_DIR}")
    print(f"\nGenerated files:")
    for config in FILES_CONFIG:
        file_name = config['name']
        print(f"  - {file_name}.csv (overwritten with averaged alignment)")
        print(f"  - {file_name}_alignment.csv (backup)")
        print(f"  - {file_name}_alignment.png/pdf")


if __name__ == "__main__":
    main()
