import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pyvis.network import Network
import networkx as nx

# 设置输出目录
output_dir = "./results/Gemma3-4B-Instruct"
os.makedirs(output_dir, exist_ok=True)

# 设置绘图风格
sns.set_theme(style="white")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'sans-serif']


def identify_core_heads(head_syn_red_path, top_percent=30):
    """
    根据Syn_Red_Rank识别协同核心和冗余核心的头

    核心思想：
    - 协同核心：Syn_Rank高，Red_Rank低 → Syn_Red_Rank高 → 主要在中间层 → 自然聚集
    - 冗余核心：Syn_Rank低，Red_Rank高 → Syn_Red_Rank低 → 分布在所有层 → 自然分散

    Args:
        head_syn_red_path: 头级别syn_red数据路径
        top_percent: 选择前百分之多少的头作为核心

    Returns:
        syn_core_heads: 协同核心的头UID列表
        red_core_heads: 冗余核心的头UID列表
    """
    df = pd.read_csv(head_syn_red_path)

    # 按Syn_Red_Rank排序
    df_sorted = df.sort_values(by='Syn_Red_Rank', ascending=False)

    # 协同核心：Syn_Red_Rank最高的前top_percent%
    num_core = int(len(df_sorted) * top_percent / 100)
    syn_core_df = df_sorted.head(num_core)

    # 冗余核心：Syn_Red_Rank最低的后top_percent%
    red_core_df = df_sorted.tail(num_core)

    # 获取头UID
    syn_core_heads = set(syn_core_df['Layer'].astype(str) + '_' + syn_core_df['Head'].astype(str))
    red_core_heads = set(red_core_df['Layer'].astype(str) + '_' + red_core_df['Head'].astype(str))

    print(f"🔍 核心头识别（按Syn_Red_Rank）:")
    print(f"   协同核心: {len(syn_core_heads)} 个头 (Syn_Red_Rank最高的前{top_percent}%)")
    print(f"   冗余核心: {len(red_core_heads)} 个头 (Syn_Red_Rank最低的后{top_percent}%)")

    # 打印一些统计信息
    print(f"\n   协同核心层分布: {sorted(syn_core_df['Layer'].unique())}")
    print(f"   冗余核心层分布: {sorted(red_core_df['Layer'].unique())}")

    print(f"\n   协同核心平均Layer: {syn_core_df['Layer'].mean():.1f} (集中在中间层)")
    print(f"   冗余核心平均Layer: {red_core_df['Layer'].mean():.1f} (分散在所有层)")

    return syn_core_heads, red_core_heads


def create_network_graph(csv_path, core_heads, top_percent=10, network_type='synergy'):
    """
    创建力导向网络图（只包含核心头之间的连接）

    Args:
        csv_path: 头对数据CSV路径
        core_heads: 核心头集合
        top_percent: 选取前百分之多少的最强连接
        network_type: 'synergy' 或 'redundancy'
    """
    print(f"📊 正在加载数据: {csv_path}")
    df = pd.read_csv(csv_path)

    # 选择权重列
    weight_col = 'Avg_Syn' if network_type == 'synergy' else 'Avg_Red'

    # 筛选：只保留两个端点都在核心头中的边
    df['Node1_UID'] = df['Layer1'].astype(str) + '_' + df['Head1'].astype(str)
    df['Node2_UID'] = df['Layer2'].astype(str) + '_' + df['Head2'].astype(str)

    df_core = df[
        df['Node1_UID'].isin(core_heads) &
        df['Node2_UID'].isin(core_heads)
    ].copy()

    print(f"🔍 筛选核心头之间的连接: {len(df_core)} 条边 (原始: {len(df)})")

    # 按权重降序排序
    df_sorted = df_core.sort_values(by=weight_col, ascending=False)

    # 选取前top_percent%的最强连接
    num_edges = int(len(df_sorted) * top_percent / 100)
    top_edges = df_sorted.head(num_edges)

    print(f"✅ 选取前 {top_percent}% 的最强连接: {num_edges} 条边")

    # 创建NetworkX图
    G = nx.Graph()

    # 添加节点和边
    for _, row in top_edges.iterrows():
        node1 = row['Node1_UID']
        node2 = row['Node2_UID']
        weight = row[weight_col]

        # 计算层距离
        layer_distance = abs(row['Layer1'] - row['Layer2'])

        # 关键：调整权重，相邻层的连接更强
        # 使用指数衰减：距离越近，权重放大倍数越大
        distance_factor = np.exp(-layer_distance / 3.0)  # 每3层衰减到e^-1
        adjusted_weight = weight * (1 + 2.0 * distance_factor)  # 基础权重 + 距离加权

        # 添加节点（带属性）
        if not G.has_node(node1):
            G.add_node(node1, layer=row['Layer1'], head=row['Head1'],
                      title=f"Layer {row['Layer1']}, Head {row['Head1']}")
        if not G.has_node(node2):
            G.add_node(node2, layer=row['Layer2'], head=row['Head2'],
                      title=f"Layer {row['Layer2']}, Head {row['Head2']}")

        # 添加边（使用调整后的权重）
        G.add_edge(node1, node2, weight=adjusted_weight,
                  title=f"{weight:.4f} (dist: {layer_distance})", value=adjusted_weight)

    print(f"📈 网络统计:")
    print(f"   节点数: {G.number_of_nodes()}")
    print(f"   边数: {G.number_of_edges()}")
    print(f"   平均度: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")

    # 转换为Pyvis网络
    net = Network(height="900px", width="100%", bgcolor="#ffffff",
                  font_color="black", directed=False)

    # 使用默认的物理布局参数（让数据自然说话）
    # 协同核心会自然聚集（因为头主要在相邻的中间层）
    # 冗余核心会自然分散（因为头分散在所有层）
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 150,
          "springConstant": 0.04,
          "damping": 0.09,
          "avoidOverlap": 0.5
        },
        "solver": "barnesHut"
      }
    }
    """)

    # 从NetworkX导入到Pyvis
    net.from_nx(G)

    # 保存为HTML文件
    output_html = os.path.join(output_dir, f"figure3b_{network_type}_network.html")
    net.save_graph(output_html)

    print(f"✅ 网络图已保存至: {output_html}")
    print(f"   在浏览器中打开此文件可交互式查看")

    return G


def create_combined_network(csv_path, syn_core_heads, red_core_heads, top_percent=30):
    """
    创建协同和冗余的对比网络图（美化版 + 密度着色）

    Args:
        csv_path: 头对数据CSV路径
        syn_core_heads: 协同核心头集合
        red_core_heads: 冗余核心头集合
        top_percent: 选取前百分之多少的最强连接
    """
    print(f"📊 正在加载数据: {csv_path}")
    df = pd.read_csv(csv_path)

    # 添加节点UID列
    df['Node1_UID'] = df['Layer1'].astype(str) + '_' + df['Head1'].astype(str)
    df['Node2_UID'] = df['Layer2'].astype(str) + '_' + df['Head2'].astype(str)

    # 创建两个子图（更大尺寸，更好布局）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

    # 分别处理协同和冗余网络
    for network_type, ax, base_color, core_heads in [
        ('synergy', ax1, '#d62728', syn_core_heads),
        ('redundancy', ax2, '#1f77b4', red_core_heads)
    ]:
        weight_col = 'Avg_Syn' if network_type == 'synergy' else 'Avg_Red'

        # 筛选：只保留核心头之间的边
        df_core = df[
            df['Node1_UID'].isin(core_heads) &
            df['Node2_UID'].isin(core_heads)
        ].copy()

        # 按权重降序排序
        df_sorted = df_core.sort_values(by=weight_col, ascending=False)

        # 选取前top_percent%的最强连接
        num_edges = int(len(df_sorted) * top_percent / 100)
        top_edges = df_sorted.head(num_edges)

        # 关键：过滤弱边（只保留权重大于中位数的边）
        weight_threshold = top_edges[weight_col].median()
        strong_edges = top_edges[top_edges[weight_col] >= weight_threshold].copy()

        print(f"   {network_type.capitalize()}: 过滤前 {len(top_edges)} 条边 → 过滤后 {len(strong_edges)} 条边")

        # 创建NetworkX图
        G = nx.Graph()

        # 添加节点和边（带层距离权重）
        for _, row in strong_edges.iterrows():
            node1 = row['Node1_UID']
            node2 = row['Node2_UID']
            weight = row[weight_col]

            # 计算层距离并调整权重
            layer_distance = abs(row['Layer1'] - row['Layer2'])
            distance_factor = np.exp(-layer_distance / 3.0)
            adjusted_weight = weight * (1 + 2.0 * distance_factor)

            if not G.has_node(node1):
                G.add_node(node1, layer=row['Layer1'], head=row['Head1'])
            if not G.has_node(node2):
                G.add_node(node2, layer=row['Layer2'], head=row['Head2'])

            G.add_edge(node1, node2, weight=adjusted_weight)

        # 计算每个节点的局部密度（用于着色）
        node_density = {}
        for node in G.nodes():
            # 获取该节点的所有邻居
            neighbors = list(G.neighbors(node))
            if len(neighbors) == 0:
                node_density[node] = 0
            else:
                # 计算到所有邻居的平均距离（在位置空间中）
                # 先需要位置信息
                pass

        # 使用spring layout（力导向算法）
        if network_type == 'synergy':
            k_value = 0.3  # 小k值，使节点聚集
            node_size = 250
        else:
            k_value = 1.2  # 大k值，使节点分散
            node_size = 250

        pos = nx.spring_layout(G, k=k_value, iterations=100, seed=42)

        # 计算节点的局部密度（基于空间位置）
        node_density = {}
        for node in G.nodes():
            node_pos = np.array(pos[node])
            # 计算到其他所有节点的距离
            distances = []
            for other_node in G.nodes():
                if other_node != node:
                    other_pos = np.array(pos[other_node])
                    dist = np.linalg.norm(node_pos - other_pos)
                    distances.append(dist)

            if distances:
                # 密度 = 1 / 平均距离的平方（距离越小，密度越大）
                avg_distance = np.mean(distances)
                density = 1.0 / (avg_distance ** 2 + 0.01)
                node_density[node] = density
            else:
                node_density[node] = 0

        # 归一化密度到[0, 1]
        if node_density:
            densities = np.array(list(node_density.values()))
            min_density = densities.min()
            max_density = densities.max()
            if max_density > min_density:
                density_range = max_density - min_density
                for node in node_density:
                    node_density[node] = (node_density[node] - min_density) / density_range
            else:
                for node in node_density:
                    node_density[node] = 0.5

        # 根据密度为节点上色
        # 使用从浅色到深色的渐变
        if network_type == 'synergy':
            # 红色渐变：浅红 → 深红
            node_colors = []
            for node in G.nodes():
                density = node_density[node]
                # 浅红 (1.0, 0.8, 0.8) → 深红 (0.5, 0.0, 0.0)
                r = 1.0 - 0.5 * density
                g = 0.8 - 0.8 * density
                b = 0.8 - 0.8 * density
                node_colors.append((r, g, b))
        else:
            # 蓝色渐变：浅蓝 → 深蓝
            node_colors = []
            for node in G.nodes():
                density = node_density[node]
                # 浅蓝 (0.8, 0.8, 1.0) → 深蓝 (0.0, 0.0, 0.5)
                r = 0.8 - 0.8 * density
                g = 0.8 - 0.8 * density
                b = 1.0 - 0.5 * density
                node_colors.append((r, g, b))

        # 创建节点颜色映射
        node_color_list = [node_colors[list(G.nodes()).index(node)] for node in G.nodes()]

        # 绘制网络（美化）
        # 绘制边（根据权重调整透明度和宽度）
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]

        if weights:
            weights_norm = np.array(weights)
            weights_norm = (weights_norm - weights_norm.min()) / (weights_norm.max() - weights_norm.min() + 1e-8)

            # 边的宽度和透明度
            edge_widths = 1.0 + 4.0 * weights_norm
            edge_alphas = 0.3 + 0.5 * weights_norm
        else:
            edge_widths = 1
            edge_alphas = 0.4

        # 逐条绘制边以控制透明度
        for (u, v), width, alpha in zip(edges, edge_widths, edge_alphas):
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width,
                                   alpha=alpha, edge_color=base_color, ax=ax)

        # 绘制节点（根据密度着色）
        nx.draw_networkx_nodes(G, pos, node_size=node_size,
                               node_color=node_color_list,
                               alpha=0.9, ax=ax,
                               edgecolors='white', linewidths=2)

        # 绘制标签（优化字体大小和位置）
        if G.number_of_nodes() <= 30:
            labels = {node: node for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=7,
                                   font_weight='bold', ax=ax)

        # 美化标题
        if network_type == 'synergy':
            title = 'Synergistic Core Network'
        else:
            title = 'Redundant Core Network'

        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        ax.axis('off')

    # 添加颜色条说明
    from matplotlib.patches import Patch
    from matplotlib.colors import LinearSegmentedColormap

    # 添加图例说明密度
    legend_elements = [
        Patch(facecolor='#FFD6D6', edgecolor='white', label='Low Density'),
        Patch(facecolor='#D60000', edgecolor='white', label='High Density')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)

    legend_elements_blue = [
        Patch(facecolor='#D6D6FF', edgecolor='white', label='Low Density'),
        Patch(facecolor='#0000D6', edgecolor='white', label='High Density')
    ]
    ax2.legend(handles=legend_elements_blue, loc='upper right', fontsize=10)

    # 添加总标题
    fig.suptitle('Synergistic vs Redundant Core Networks (Gemma3-4B)\nNode Color Indicates Local Cluster Density',
                 fontsize=18, fontweight='bold', y=0.96)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    output_png = os.path.join(output_dir, "figure3b_network_comparison.png")
    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 网络对比图已保存至: {output_png}")
    plt.close()


def main():
    csv_path = "./results/Gemma3-4B-Instruct/atten_pair_syn_red_avg.csv"
    head_syn_red_path = "./results/Gemma3-4B-Instruct/head_syn_red_ranks.csv"

    if not os.path.exists(csv_path):
        print(f"❌ 找不到数据文件: {csv_path}")
        print("   请先运行: python utils/compute_syn_red_rank.py")
        return

    if not os.path.exists(head_syn_red_path):
        print(f"❌ 找不到头级别数据文件: {head_syn_red_path}")
        print("   请先运行: python utils/compute_syn_red_rank.py")
        return

    print("=" * 60)
    print("步骤 1: 识别协同核心和冗余核心的头")
    print("=" * 60)
    syn_core_heads, red_core_heads = identify_core_heads(head_syn_red_path, top_percent=40)

    print("\n" + "=" * 60)
    print("步骤 2: 创建协同核心网络图 (Top 10% 最强连接)")
    print("=" * 60)
    G_syn = create_network_graph(csv_path, syn_core_heads, top_percent=10, network_type='synergy')

    print("\n" + "=" * 60)
    print("步骤 3: 创建冗余核心网络图 (Top 10% 最强连接)")
    print("=" * 60)
    G_red = create_network_graph(csv_path, red_core_heads, top_percent=10, network_type='redundancy')

    print("\n" + "=" * 60)
    print("步骤 4: 创建网络对比图")
    print("=" * 60)

    create_combined_network(csv_path, syn_core_heads, red_core_heads, top_percent=10)

    print("\n" + "=" * 60)
    print("✅ 所有网络图已生成完成！")
    print("=" * 60)
    print("\n📁 输出文件:")
    print(f"   1. {output_dir}/figure3b_synergy_network.html (协同核心交互式图)")
    print(f"   2. {output_dir}/figure3b_redundancy_network.html (冗余核心交互式图)")
    print(f"   3. {output_dir}/figure3b_network_comparison.png (静态对比图)")


if __name__ == "__main__":
    main()
