import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from community import community_louvain
import os

# 设置输出目录
output_dir = "./results/Gemma3-4B-Instruct"
os.makedirs(output_dir, exist_ok=True)

# 设置绘图风格
sns.set_theme(style="white")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'sans-serif']


def create_adjacency_matrix(pair_df, all_head_uids, weight_col, invert_for_distance=False, epsilon=1e-6):
    """
    创建邻接矩阵

    论文方法：
    - 创建 N×N 邻接矩阵
    - A[i,j] = 头i和头j之间的Syn或Red值

    重要：在加权图中计算全局效率时，距离 = 权重的倒数
    - 权重越大 → 连接越紧密 → 距离越短
    - distance = 1 / (Syn或Red值)

    Args:
        pair_df: 头对数据
        all_head_uids: 所有头UID列表
        weight_col: 权重列名（'Avg_Syn' 或 'Avg_Red'）
        invert_for_distance: 是否反转权重用于计算距离
        epsilon: 小常数，避免除以0

    Returns:
        adj_matrix: 邻接矩阵（numpy array）
        G: NetworkX图
    """
    n = len(all_head_uids)
    adj_matrix = np.zeros((n, n))

    # 创建UID到索引的映射
    uid_to_idx = {uid: idx for idx, uid in enumerate(all_head_uids)}

    # 构建邻接矩阵
    num_edges = 0
    for _, row in pair_df.iterrows():
        node1 = row['Node1_UID']
        node2 = row['Node2_UID']
        weight = row[weight_col]

        if node1 in uid_to_idx and node2 in uid_to_idx:
            idx1, idx2 = uid_to_idx[node1], uid_to_idx[node2]

            if invert_for_distance:
                # 权重反转：用于计算全局效率
                # 权重越大 → 连接越紧密 → 距离越短
                distance = 1.0 / (weight + epsilon)
                adj_matrix[idx1, idx2] = distance
                adj_matrix[idx2, idx1] = distance
            else:
                # 直接使用原始权重：用于模块性计算
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

    # 创建NetworkX图
    G = nx.from_numpy_array(adj_matrix)

    return adj_matrix, G


def calculate_global_efficiency(G):
    """
    计算网络的全局效率（加权版本）

    公式：E_global = 1/[n(n-1)] * Σ[i≠j] 1/d(i,j)

    其中 d(i,j) 是加权最短路径距离
    - 使用边权重作为距离
    - 权重越小 → 距离越短 → 效率越高

    Args:
        G: NetworkX图（边权重 = 1/原始连接强度）

    Returns:
        global_efficiency: 全局效率值
    """
    n = len(G.nodes())
    if n <= 1:
        return 0.0

    total_efficiency = 0.0
    nodes = list(G.nodes())

    for i in range(n):
        for j in range(i + 1, n):
            try:
                # 使用边权重作为距离计算最短路径
                path_length = nx.shortest_path_length(G, nodes[i], nodes[j], weight='weight')
                if path_length > 0:
                    efficiency = 1.0 / path_length
                    total_efficiency += efficiency
            except:
                pass

    # 归一化
    max_possible = n * (n - 1) / 2
    if max_possible > 0:
        return total_efficiency / max_possible
    return 0.0


def calculate_modularity(G):
    """
    计算网络的模块性

    公式：Q = 1/(2m) * Σ[A_ij - k_i*k_j/(2m)] * δ(c_i, c_j)

    Args:
        G: NetworkX图

    Returns:
        modularity: 模块性值
        num_communities: 社区数量
    """
    try:
        # 使用Louvain算法检测社区
        partition = community_louvain.best_partition(G, weight='weight')

        # 计算模块性
        modularity = community_louvain.modularity(partition, G, weight='weight')

        # 统计社区数量
        num_communities = len(set(partition.values()))

        return modularity, num_communities
    except Exception as e:
        print(f"   Warning: Could not calculate modularity: {e}")
        return 0.0, 1


def analyze_networks():
    """
    使用论文方法构建网络并计算全局效率和模块性

    论文方法：
    1. 创建 N×N 邻接矩阵，A[i,j] = Syn或Red值
    2. 全局效率：距离 = 权重的倒数（权重越大，距离越短）
    3. 模块性：直接使用原始权重

    Returns:
        results: 分析结果字典
    """
    # 读取数据
    head_df = pd.read_csv("./results/Gemma3-4B-Instruct/head_syn_red_ranks.csv")
    pair_df = pd.read_csv("./results/Gemma3-4B-Instruct/atten_pair_syn_red_avg.csv")

    # 添加节点UID列
    pair_df['Node1_UID'] = pair_df['Layer1'].astype(str) + '_' + pair_df['Head1'].astype(str)
    pair_df['Node2_UID'] = pair_df['Layer2'].astype(str) + '_' + pair_df['Head2'].astype(str)

    # 获取所有头UID（排序后）
    all_head_uids = sorted(set(head_df['Layer'].astype(str) + '_' + head_df['Head'].astype(str)))

    print(f"{'=' * 60}")
    print(f"复现 Figure 3c: 网络性质对比")
    print(f"节点数: {len(all_head_uids)} (全部注意力头)")
    print(f"方法: 论文方法")
    print(f"  - 全局效率: 距离 = 1/权重，权重越大距离越短")
    print(f"  - 模块性: 直接使用原始权重")
    print(f"{'=' * 60}")

    # 打印原始权重范围
    print(f"\n原始权重范围:")
    print(f"   Avg_Syn:  [{pair_df['Avg_Syn'].min():.6f}, {pair_df['Avg_Syn'].max():.6f}]")
    print(f"   Avg_Red:  [{pair_df['Avg_Red'].min():.6f}, {pair_df['Avg_Red'].max():.6f}]")

    results = {}

    # 分析协同网络
    print(f"\n{'=' * 60}")
    print("分析协同网络")
    print(f"{'=' * 60}")

    # 为全局效率创建反转权重的图
    print("\n1. 创建距离图（权重反转，用于全局效率）:")
    _, G_syn_distance = create_adjacency_matrix(pair_df, all_head_uids, 'Avg_Syn',
                                                invert_for_distance=True)
    syn_eff = calculate_global_efficiency(G_syn_distance)

    # 为模块性创建原始权重的图
    print("\n2. 创建原始权重图（用于模块性）:")
    _, G_syn_raw = create_adjacency_matrix(pair_df, all_head_uids, 'Avg_Syn',
                                           invert_for_distance=False)
    syn_mod, syn_comms = calculate_modularity(G_syn_raw)

    results['synergy'] = {
        'global_efficiency': syn_eff,
        'modularity': syn_mod,
        'num_communities': syn_comms,
        'num_nodes': G_syn_distance.number_of_nodes(),
        'num_edges': G_syn_distance.number_of_edges()
    }

    print(f"\n结果:")
    print(f"   节点数: {results['synergy']['num_nodes']}")
    print(f"   边数: {results['synergy']['num_edges']}")
    print(f"   全局效率: {syn_eff:.4f}")
    print(f"   模块性: {syn_mod:.4f}")
    print(f"   社区数: {syn_comms}")

    # 分析冗余网络
    print(f"\n{'=' * 60}")
    print("分析冗余网络")
    print(f"{'=' * 60}")

    # 为全局效率创建反转权重的图
    print("\n1. 创建距离图（权重反转，用于全局效率）:")
    _, G_red_distance = create_adjacency_matrix(pair_df, all_head_uids, 'Avg_Red',
                                                invert_for_distance=True)
    red_eff = calculate_global_efficiency(G_red_distance)

    # 为模块性创建原始权重的图
    print("\n2. 创建原始权重图（用于模块性）:")
    _, G_red_raw = create_adjacency_matrix(pair_df, all_head_uids, 'Avg_Red',
                                           invert_for_distance=False)
    red_mod, red_comms = calculate_modularity(G_red_raw)

    results['redundancy'] = {
        'global_efficiency': red_eff,
        'modularity': red_mod,
        'num_communities': red_comms,
        'num_nodes': G_red_distance.number_of_nodes(),
        'num_edges': G_red_distance.number_of_edges()
    }

    print(f"\n结果:")
    print(f"   节点数: {results['redundancy']['num_nodes']}")
    print(f"   边数: {results['redundancy']['num_edges']}")
    print(f"   全局效率: {red_eff:.4f}")
    print(f"   模块性: {red_mod:.4f}")
    print(f"   社区数: {red_comms}")

    return results


def plot_figure3c(results):
    """
    绘制Figure 3c：全局效率和模块性的对比条形图

    Args:
        results: 分析结果字典
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 数据
    metrics = ['Global Efficiency', 'Modularity']
    syn_values = [results['synergy']['global_efficiency'], results['synergy']['modularity']]
    red_values = [results['redundancy']['global_efficiency'], results['redundancy']['modularity']]

    x = np.arange(len(metrics))
    width = 0.35

    # 绘制条形图
    bars1 = ax1.bar(x - width/2, syn_values, width, label='Synergistic Network',
                    color='#d62728', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, red_values, width, label='Redundant Network',
                    color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=1.5)

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))

    ax1.set_ylabel('Value', fontsize=13, fontweight='bold')
    ax1.set_title('Network Metrics Comparison (Gemma3-4B)',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax1.set_axisbelow(True)

    # 绘制对比图（雷达图风格）
    categories = ['Global\nEfficiency', 'Modularity']

    # 归一化到0-1以便比较
    max_eff = max(syn_values[0], red_values[0])
    max_mod = max(syn_values[1], red_values[1])

    # 避免除以0
    if max_eff == 0:
        max_eff = 1
    if max_mod == 0:
        max_mod = 1

    syn_normalized = [syn_values[0]/max_eff, syn_values[1]/max_mod]
    red_normalized = [red_values[0]/max_eff, red_values[1]/max_mod]

    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    syn_normalized += syn_normalized[:1]
    red_normalized += red_normalized[:1]
    angles += angles[:1]

    ax2 = plt.subplot(122, projection='polar')
    ax2.plot(angles, syn_normalized, 'o-', linewidth=2.5, color='#d62728',
             label='Synergistic Network', markersize=8)
    ax2.fill(angles, syn_normalized, alpha=0.15, color='#d62728')
    ax2.plot(angles, red_normalized, 's-', linewidth=2.5, color='#1f77b4',
             label='Redundant Network', markersize=8)
    ax2.fill(angles, red_normalized, alpha=0.15, color='#1f77b4')

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax2.set_title('Normalized Comparison', fontsize=14, fontweight='bold', pad=15)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "figure3c_efficiency_modularity.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Figure 3c已保存至: {output_path}")
    plt.close()

    # 打印关键发现
    print("\n" + "=" * 60)
    print("📊 关键发现")
    print("=" * 60)

    eff_diff = syn_values[0] - red_values[0]
    mod_diff = red_values[1] - syn_values[1]

    # 计算相对差异（避免除以0）
    if red_values[0] > 0:
        eff_percent = eff_diff/red_values[0]*100
    else:
        eff_percent = 0

    if syn_values[1] > 0:
        mod_percent = mod_diff/syn_values[1]*100
    else:
        mod_percent = 0

    print(f"✅ 协同网络的全局效率({syn_values[0]:.4f}) vs 冗余网络({red_values[0]:.4f})")
    print(f"   差异: {eff_diff:.4f} ({eff_percent:.1f}%)")

    print(f"\n✅ 冗余网络的模块性({red_values[1]:.4f}) vs 协同网络({syn_values[1]:.4f})")
    print(f"   差异: {mod_diff:.4f} ({mod_percent:.1f}%)")

    print(f"\n🔍 验证论文发现:")
    if syn_values[0] > red_values[0]:
        print(f"   ✓ 协同网络表现出更高的全局效率")
    else:
        print(f"   ✗ 未验证：协同网络全局效率较低或相等")

    if red_values[1] > syn_values[1]:
        print(f"   ✓ 冗余网络表现出更高的模块性")
    else:
        print(f"   ✗ 未验证：冗余网络模块性较低或相等")


def main():
    print("=" * 60)
    print("复现 Figure 3c: 网络性质对比")
    print("=" * 60)

    # 使用论文方法
    results = analyze_networks()

    # 绘制Figure 3c
    print("\n" + "=" * 60)
    print("生成Figure 3c可视化")
    print("=" * 60)
    plot_figure3c(results)

    # 保存结果到CSV
    results_df = pd.DataFrame([{
        'Network_Type': 'Synergistic',
        'Global_Efficiency': results['synergy']['global_efficiency'],
        'Modularity': results['synergy']['modularity'],
        'Num_Communities': results['synergy']['num_communities'],
        'Num_Nodes': results['synergy']['num_nodes'],
        'Num_Edges': results['synergy']['num_edges']
    }, {
        'Network_Type': 'Redundant',
        'Global_Efficiency': results['redundancy']['global_efficiency'],
        'Modularity': results['redundancy']['modularity'],
        'Num_Communities': results['redundancy']['num_communities'],
        'Num_Nodes': results['redundancy']['num_nodes'],
        'Num_Edges': results['redundancy']['num_edges']
    }])

    csv_path = os.path.join(output_dir, "figure3c_metrics.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n✅ 数据已保存至: {csv_path}")

    print("\n" + "=" * 60)
    print("✅ 分析完成！")
    print("=" * 60)
    print(f"\n💡 论文方法:")
    print(f"   - 邻接矩阵: A[i,j] = Syn或Red值")
    print(f"   - 全局效率: 距离 = 1/权重（权重越大，距离越短）")
    print(f"   - 模块性: 直接使用原始权重")


if __name__ == "__main__":
    main()
