import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from community import community_louvain
import os
from tqdm import tqdm

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
_COLOR_SYN = '#E64B35'         # warm red
_COLOR_RED = '#4DBBD5'         # teal


def create_adjacency_matrix(pair_df, all_head_uids, weight_col, invert_for_distance=False, epsilon=1e-6):
    """
    创建邻接矩阵
    """
    n = len(all_head_uids)
    adj_matrix = np.zeros((n, n))

    uid_to_idx = {uid: idx for idx, uid in enumerate(all_head_uids)}

    num_edges = 0
    for _, row in pair_df.iterrows():
        node1 = row['Node1_UID']
        node2 = row['Node2_UID']
        weight = row[weight_col]

        if node1 in uid_to_idx and node2 in uid_to_idx:
            idx1, idx2 = uid_to_idx[node1], uid_to_idx[node2]

            if invert_for_distance:
                distance = 1.0 / (weight + epsilon)
                adj_matrix[idx1, idx2] = distance
                adj_matrix[idx2, idx1] = distance
            else:
                adj_matrix[idx1, idx2] = weight
                adj_matrix[idx2, idx1] = weight
            num_edges += 1

    print(f"   邻接矩阵大小: {n}×{n}")
    print(f"   非零边数: {num_edges}")

    if invert_for_distance:
        print(f"   原始权重范围: [{pair_df[weight_col].min():.6f}, {pair_df[weight_col].max():.6f}]")
        print(f"   距离范围（1/权重）: [{adj_matrix[adj_matrix > 0].min():.6f}, {adj_matrix[adj_matrix > 0].max():.6f}]")
    else:
        print(f"   权重范围: [{adj_matrix[adj_matrix > 0].min():.6f}, {adj_matrix[adj_matrix > 0].max():.6f}]")

    G = nx.from_numpy_array(adj_matrix)
    return adj_matrix, G


def calculate_global_efficiency(G, sample_ratio=0.001):
    """计算网络的全局效率（采样版本）"""
    n = len(G.nodes())
    if n <= 1:
        return 0.0

    nodes = list(G.nodes())

    # 采样0.1%的节点对
    sample_size = max(100, int(n * (n - 1) / 2 * sample_ratio))
    print(f"   采样比例: {sample_ratio*100}%, 采样节点对数: {sample_size}")

    # 随机采样节点对
    import random
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    sampled_pairs = random.sample(all_pairs, min(sample_size, len(all_pairs)))

    total_efficiency = 0.0
    for i, j in tqdm(sampled_pairs, desc="   计算全局效率"):
        try:
            path_length = nx.shortest_path_length(G, nodes[i], nodes[j], weight='weight')
            if path_length > 0:
                efficiency = 1.0 / path_length
                total_efficiency += efficiency
        except:
            pass

    return total_efficiency / len(sampled_pairs) if sampled_pairs else 0.0


def calculate_modularity(G):
    """计算网络的模块性"""
    try:
        partition = community_louvain.best_partition(G, weight='weight')
        modularity = community_louvain.modularity(partition, G, weight='weight')
        num_communities = len(set(partition.values()))
        return modularity, num_communities
    except Exception as e:
        print(f"   Warning: Could not calculate modularity: {e}")
        return 0.0, 1


def analyze_networks(csv_path):
    """分析网络并计算指标"""
    print(f"正在加载数据: {csv_path}")
    df = pd.read_csv(csv_path)

    # 适配列名
    df['Layer1'] = df['layer_1']
    df['Head1'] = df['head_1']
    df['Layer2'] = df['layer_2']
    df['Head2'] = df['head_2']

    # 对10个问题求平均
    print("对10个问题求平均...")
    df_avg = df.groupby(['Layer1', 'Head1', 'Layer2', 'Head2']).agg({
        'syn': 'mean',
        'red': 'mean'
    }).reset_index()

    df_avg['Avg_Syn'] = df_avg['syn']
    df_avg['Avg_Red'] = df_avg['red']

    # 添加节点UID列
    df_avg['Node1_UID'] = df_avg['Layer1'].astype(str) + '_' + df_avg['Head1'].astype(str)
    df_avg['Node2_UID'] = df_avg['Layer2'].astype(str) + '_' + df_avg['Head2'].astype(str)

    # 获取所有头UID
    all_heads_1 = set(zip(df_avg['Layer1'], df_avg['Head1']))
    all_heads_2 = set(zip(df_avg['Layer2'], df_avg['Head2']))
    all_heads = all_heads_1 | all_heads_2
    all_head_uids = sorted([f"{l}_{h}" for l, h in all_heads])

    print(f"节点数: {len(all_head_uids)}")
    print(f"边数: {len(df_avg)}")

    results = {}

    # 分析协同网络
    print(f"\n{'=' * 60}")
    print("分析协同网络 (Synergy)")
    print(f"{'=' * 60}")

    print("\n1. 创建距离图（用于全局效率）:")
    _, G_syn_distance = create_adjacency_matrix(df_avg, all_head_uids, 'Avg_Syn',
                                            invert_for_distance=True)
    syn_eff = calculate_global_efficiency(G_syn_distance)

    print("\n2. 创建原始权重图（用于模块性）:")
    _, G_syn_raw = create_adjacency_matrix(df_avg, all_head_uids, 'Avg_Syn',
                                   invert_for_distance=False)
    syn_mod, syn_comms = calculate_modularity(G_syn_raw)

    results['synergy'] = {
        'global_efficiency': syn_eff,
        'modularity': syn_mod,
        'num_communities': syn_comms,
    }

    print(f"\n协同网络结果:")
    print(f"   全局效率: {syn_eff:.4f}")
    print(f"   模块性: {syn_mod:.4f}")
    print(f"   社区数: {syn_comms}")

    # 分析冗余网络
    print(f"\n{'=' * 60}")
    print("分析冗余网络 (Redundancy)")
    print(f"{'=' * 60}")

    print("\n1. 创建距离图（用于全局效率）:")
    _, G_red_distance = create_adjacency_matrix(df_avg, all_head_uids, 'Avg_Red',
                                            invert_for_distance=True)
    red_eff = calculate_global_efficiency(G_red_distance)

    print("\n2. 创建原始权重图（用于模块性）:")
    _, G_red_raw = create_adjacency_matrix(df_avg, all_head_uids, 'Avg_Red',
                                   invert_for_distance=False)
    red_mod, red_comms = calculate_modularity(G_red_raw)

    results['redundancy'] = {
        'global_efficiency': red_eff,
        'modularity': red_mod,
        'num_communities': red_comms,
    }

    print(f"\n冗余网络结果:")
    print(f"   全局效率: {red_eff:.4f}")
    print(f"   模块性: {red_mod:.4f}")
    print(f"   社区数: {red_comms}")

    return results


def plot_global_efficiency(results, output_path):
    """Plot global efficiency comparison."""
    fig, ax = plt.subplots(figsize=(7, 6.5))

    networks = ['Synergistic\nNetwork', 'Redundant\nNetwork']
    values = [results['synergy']['global_efficiency'], results['redundancy']['global_efficiency']]
    colors = [_COLOR_SYN, _COLOR_RED]

    bars = ax.bar(networks, values, width=0.52, color=colors, alpha=0.90, edgecolor='none')

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=20, fontweight='bold',
                color='#333333')

    ax.set_ylabel('Global Efficiency')
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(True, axis='y', linestyle='-', alpha=0.10, linewidth=0.5, color='#666666')
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', length=5)
    ax.tick_params(axis='x', labelsize=22)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()


def plot_modularity(results, output_path):
    """Plot modularity comparison."""
    fig, ax = plt.subplots(figsize=(7, 6.5))

    networks = ['Synergistic\nNetwork', 'Redundant\nNetwork']
    values = [results['synergy']['modularity'], results['redundancy']['modularity']]
    colors = [_COLOR_SYN, _COLOR_RED]

    bars = ax.bar(networks, values, width=0.52, color=colors, alpha=0.90, edgecolor='none')

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=20, fontweight='bold',
                color='#333333')

    ax.set_ylabel('Modularity')
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(True, axis='y', linestyle='-', alpha=0.10, linewidth=0.5, color='#666666')
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', length=5)
    ax.tick_params(axis='x', labelsize=22)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    # 配置
    INPUT_FILE = "/root/data1/zjj/Neurlps2026/Results/Llama-3.1-8B-Instruct/data/pairwise/al_syn_red_pairwise.csv"
    OUTPUT_DIR = "/root/data1/zjj/Neurlps2026/Results/Plots/Llama-3.1-8B-Instruct"
    OUTPUT_EFF = os.path.join(OUTPUT_DIR, "global_efficiency.png")
    OUTPUT_MOD = os.path.join(OUTPUT_DIR, "modularity.png")

    print("=" * 60)
    print("Gemma3-4B-Instruct GSM8K Network Metrics Analysis")
    print("=" * 60)
    print(f"\n输入: {INPUT_FILE}")

    if not os.path.exists(INPUT_FILE):
        print(f"\n错误: 找不到输入文件: {INPUT_FILE}")
        exit(1)

    # 分析网络
    results = analyze_networks(INPUT_FILE)

    # 绘图
    print(f"\n{'=' * 60}")
    print("生成图表")
    print(f"{'=' * 60}")

    plot_global_efficiency(results, OUTPUT_EFF)
    plot_modularity(results, OUTPUT_MOD)

    print(f"\n{'=' * 60}")
    print("完成！")
    print(f"{'=' * 60}")
    print(f"\n输出文件:")
    print(f"  - {OUTPUT_EFF}")
    print(f"  - {OUTPUT_MOD}")

    # 打印关键发现
    print(f"\n关键发现:")
    print(f"  协同网络全局效率: {results['synergy']['global_efficiency']:.4f}")
    print(f"  冗余网络全局效率: {results['redundancy']['global_efficiency']:.4f}")
    print(f"  协同网络模块性: {results['synergy']['modularity']:.4f}")
    print(f"  冗余网络模块性: {results['redundancy']['modularity']:.4f}")
