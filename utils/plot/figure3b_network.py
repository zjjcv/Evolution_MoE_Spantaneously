import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

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
_COLOR_SYN = '#E64B35'         # warm red for synergy
_COLOR_RED = '#4DBBD5'         # teal for redundancy


def create_network_graph(csv_path, output_path, top_percent=10, weight_type='syn'):
    """
    Create undirected network graph based on syn or red weights.

    Args:
        csv_path: Path to pairwise syn/red CSV data
        output_path: Path to save the output figure
        top_percent: Percentage of top connections to display (0-100)
        weight_type: 'syn' for synergy or 'red' for redundancy
    """
    print(f"Loading data: {csv_path}")
    df = pd.read_csv(csv_path)

    # Adapt to new data column names
    df['Layer1'] = df['layer_1']
    df['Head1'] = df['head_1']
    df['Layer2'] = df['layer_2']
    df['Head2'] = df['head_2']

    # Average across questions: group by head pairs
    print("Averaging across questions...")
    df_avg = df.groupby(['Layer1', 'Head1', 'Layer2', 'Head2']).agg({
        'syn': 'mean',
        'red': 'mean'
    }).reset_index()

    df_avg['Avg_Syn'] = df_avg['syn']
    df_avg['Avg_Red'] = df_avg['red']
    df = df_avg

    print(f"Unique head pairs: {len(df_avg)}")

    # Select weight column
    weight_col = 'Avg_Syn' if weight_type == 'syn' else 'Avg_Red'

    # Add node UID columns
    df['Node1_UID'] = df['Layer1'].astype(str) + '_' + df['Head1'].astype(str)
    df['Node2_UID'] = df['Layer2'].astype(str) + '_' + df['Head2'].astype(str)

    # Sort by weight descending
    df_sorted = df.sort_values(by=weight_col, ascending=False)

    # Select top top_percent% strongest connections
    num_edges = int(len(df_sorted) * top_percent / 100)
    top_edges = df_sorted.head(num_edges)

    print(f"Total edges: {len(df)}, selecting top {top_percent}%: {num_edges} edges")

    # Create NetworkX graph
    G = nx.Graph()

    # Add nodes and edges
    for _, row in top_edges.iterrows():
        node1 = row['Node1_UID']
        node2 = row['Node2_UID']
        weight = row[weight_col]

        # Calculate layer distance
        layer_distance = abs(row['Layer1'] - row['Layer2'])

        # Adjust weight: stronger connections for adjacent layers
        # Use exponential decay: closer layers get amplified weights
        distance_factor = np.exp(-layer_distance / 3.0)
        adjusted_weight = weight * (1 + 2.0 * distance_factor)

        # Add nodes with attributes
        if not G.has_node(node1):
            G.add_node(node1, layer=row['Layer1'], head=row['Head1'],
                      title=f"Layer {row['Layer1']}, Head {row['Head1']}")
        if not G.has_node(node2):
            G.add_node(node2, layer=row['Layer2'], head=row['Head2'],
                      title=f"Layer {row['Layer2']}, Head {row['Head2']}")

        # Add edge with adjusted weight
        G.add_edge(node1, node2, weight=adjusted_weight,
                  title=f"{weight:.4f} (dist: {layer_distance})", value=adjusted_weight)

    print(f"Network statistics:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Avg degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")

    # Create figure with publication-quality size
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))

    # Use spring layout (force-directed algorithm)
    k_value = 0.8
    pos = nx.spring_layout(G, k=k_value, iterations=100, seed=42)

    # Calculate local density for each node
    node_density = {}
    for node in G.nodes():
        node_pos = np.array(pos[node])
        distances = []
        for other_node in G.nodes():
            if other_node != node:
                other_pos = np.array(pos[other_node])
                dist = np.linalg.norm(node_pos - other_pos)
                distances.append(dist)

        if distances:
            avg_distance = np.mean(distances)
            density = 1.0 / (avg_distance ** 2 + 0.01)
            node_density[node] = density
        else:
            node_density[node] = 0

    # Normalize density to [0, 1]
    densities = np.array(list(node_density.values()))
    min_density = densities.min()
    max_density = densities.max()
    if max_density > min_density:
        density_range = max_density - min_density
        for node in node_density:
            node_density[node] = (node_density[node] - min_density) / density_range

    # Color nodes by density using publication-quality colors
    node_colors = []
    base_color = _COLOR_SYN if weight_type == 'syn' else _COLOR_RED

    if weight_type == 'syn':
        # Red gradient: light → dark
        for node in G.nodes():
            density = node_density[node]
            r = 0.90 - 0.40 * density
            g = 0.29 + 0.20 * density
            b = 0.21 + 0.20 * density
            node_colors.append((r, g, b))
    else:
        # Teal gradient: light → dark
        for node in G.nodes():
            density = node_density[node]
            r = 0.30 + 0.20 * density
            g = 0.74 - 0.35 * density
            b = 0.83 - 0.45 * density
            node_colors.append((r, g, b))

    # Draw edges (adjust width and alpha by weight - wider range for gradual fading)
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]

    if weights:
        weights_norm = np.array(weights)
        weights_norm = (weights_norm - weights_norm.min()) / (weights_norm.max() - weights_norm.min() + 1e-8)

        # Wider ranges for more dramatic visual hierarchy
        edge_widths = 0.3 + 4.0 * weights_norm
        # Gradual fading: very transparent (0.08) for weak edges, solid (0.85) for strong
        edge_alphas = 0.08 + 0.77 * weights_norm
    else:
        edge_widths = 0.3
        edge_alphas = 0.15

    # Draw edges with individual alpha control (sorted: draw weak edges first)
    edge_color = _COLOR_SYN if weight_type == 'syn' else _COLOR_RED
    edge_draw_order = sorted(zip(edges, edge_widths, edge_alphas), key=lambda x: x[2])
    for (u, v), width, alpha in edge_draw_order:
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width,
                               alpha=alpha, edge_color=edge_color, ax=ax)

    # Calculate node importance based on:
    # 1. Local density (spatial clustering)
    # 2. Total edge weight connected to node (network centrality proxy)
    node_importance = {}
    for node in G.nodes():
        density = node_density[node]
        # Sum of edge weights for this node
        connected_weight = sum(G[node][neighbor]['weight'] for neighbor in G.neighbors(node))
        node_importance[node] = density * 0.6 + (connected_weight / max(1, max(w for w in [G[u][v]['weight'] for u, v in G.edges()]))) * 0.4

    # Normalize importance to [0, 1]
    importances = np.array(list(node_importance.values()))
    if importances.max() > importances.min():
        for node in node_importance:
            node_importance[node] = (node_importance[node] - importances.min()) / (importances.max() - importances.min())

    # Draw nodes with individual alpha based on importance (gradual fading)
    node_alphas = []
    node_sizes = []
    for node in G.nodes():
        imp = node_importance[node]
        # Low importance nodes are more transparent and smaller
        alpha = 0.25 + 0.70 * imp  # Range: 0.25 to 0.95
        size = 250 + 280 * imp      # Range: 250 to 530
        node_alphas.append(alpha)
        node_sizes.append(size)

    # Draw nodes with individual alpha and size (sorted: draw weak nodes first)
    node_draw_order = sorted(zip(G.nodes(), node_colors, node_sizes, node_alphas),
                             key=lambda x: x[3])
    for node, color, size, alpha in node_draw_order:
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_size=size,
                               node_color=[color],
                               alpha=alpha, ax=ax,
                               edgecolors='white', linewidths=2.0)

    # Draw labels for smaller networks
    if G.number_of_nodes() <= 50:
        labels = {node: node for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=7,
                               font_weight='bold', ax=ax)

    # No title - leave it clean for publication
    ax.axis('off')

    # Add color bar legend (larger, bold font)
    from matplotlib.patches import Patch

    if weight_type == 'syn':
        legend_elements = [
            Patch(facecolor='#E68A7F', edgecolor='white', label='Low Density'),
            Patch(facecolor='#A3140B', edgecolor='white', label='High Density')
        ]
    else:
        legend_elements = [
            Patch(facecolor='#A3D9E8', edgecolor='white', label='Low Density'),
            Patch(facecolor='#006D8F', edgecolor='white', label='High Density')
        ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=28, prop={'weight': 'bold'},
              framealpha=0.98, edgecolor='#CCCCCC', fancybox=False)

    plt.tight_layout()

    # Create output directory and save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    # Gemma3-4B-Instruct GSM8K data configuration
    INPUT_FILE = "/root/data1/zjj/Neurlps2026/Results/Llama-3.1-8B-Instruct/data/pairwise/al_syn_red_pairwise.csv"
    OUTPUT_DIR = "/root/data1/zjj/Neurlps2026/Results/Plots/Llama-3.1-8B-Instruct"

    print("=" * 60)
    print("Gemma3-4B-Instruct GSM8K Network Graph Generator")
    print("=" * 60)
    print(f"\nInput:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_DIR}")

    if not os.path.exists(INPUT_FILE):
        print(f"\nError: Input file not found: {INPUT_FILE}")
        exit(1)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate output paths
    output_syn = os.path.join(OUTPUT_DIR, "gemma3_gsm8k_syn_network.png")
    output_red = os.path.join(OUTPUT_DIR, "gemma3_gsm8k_red_network.png")

    # Plot synergy network (top 3% edges)
    print(f"\n{'='*60}")
    print("Plotting Synergy Network (top 3% edges)")
    print(f"{'='*60}")
    create_network_graph(INPUT_FILE, output_syn, top_percent=3.0, weight_type='syn')

    # Plot redundancy network (top 3% edges)
    print(f"\n{'='*60}")
    print("Plotting Redundancy Network (top 3% edges)")
    print(f"{'='*60}")
    create_network_graph(INPUT_FILE, output_red, top_percent=3.0, weight_type='red')

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files (2 figures):")
    print(f"  - gemma3_gsm8k_syn_network.png")
    print(f"  - gemma3_gsm8k_red_network.png")
