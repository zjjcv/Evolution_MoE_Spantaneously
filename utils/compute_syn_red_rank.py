import os
import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 配置参数
# ---------------------------------------------------------
# Gemma3配置：6个prompts，完整PhiID计算
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 计算量：34层×8头=272个头 → 37,000对头组合 × 6 prompts = 222,000次计算
# 预计时间：约30-40分钟（16进程，完整PhiID）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROMPT_SAMPLE_FRACTION = 6 / 6  # Gemma3使用6个prompts
N_WORKERS = 16  # 多进程数
USE_FAST_APPROXIMATION = False  # 使用完整PhiID计算（与论文一致）

# ---------------------------------------------------------
# 导入完整 PhiID 库
# ---------------------------------------------------------
try:
    from phyid.calculate import calc_PhiID
    print("✅ 使用完整 PhiID 计算（与论文一致）")
except ImportError:
    print("⚠️ 警告：无法找到 phyid 模块。请先安装 integrated-info-decomp 库：")
    print("   pip install -e /path/to/integrated-info-decomp/")
    raise


# ============================================================
# 快速近似算法（基于相关性，改进版）
# ============================================================
def fast_correlation_syn_red(ts1, ts2, tau=1, n_samples=50):
    """
    快速近似计算协同和冗余（改进版）

    核心优化：
    1. 时间点采样：从99个点减少到50个点（增加准确性）
    2. 改进的协同-冗余计算公式，更接近完整PhiID

    Args:
        ts1, ts2: 时间序列
        tau: 时间滞后
        n_samples: 采样点数

    Returns:
        (syn_syn, red_red) 元组
    """
    # 1. 时间点采样（均匀采样）
    n = len(ts1)
    if n > n_samples:
        indices = np.linspace(0, n - 1, n_samples, dtype=int)
        ts1 = ts1[indices]
        ts2 = ts2[indices]
        n = n_samples

    # 2. 构建过去和未来状态
    src_past, src_future = ts1[:-tau], ts1[tau:]
    trg_past, trg_future = ts2[:-tau], ts2[tau:]

    # 3. 计算关键相关性
    def safe_corr(x, y):
        if len(x) < 2:
            return 0.0
        r = np.corrcoef(x, y)[0, 1]
        return np.nan_to_num(r) if not np.isnan(r) else 0.0

    r_xx = safe_corr(src_past, src_future)
    r_yy = safe_corr(trg_past, trg_future)
    r_xy_past = safe_corr(src_past, trg_past)
    r_xy_future = safe_corr(src_future, trg_future)
    r_xp_yf = safe_corr(src_past, trg_future)
    r_yp_xf = safe_corr(trg_past, src_future)

    # 4. 转换为互信息近似：I(X;Y) ≈ -0.5 * log(1 - r²)
    def corr_to_mi(r):
        r_abs = min(abs(r), 0.999)
        return -0.5 * np.log(1 - r_abs**2)

    i_xx = corr_to_mi(r_xx)
    i_yy = corr_to_mi(r_yy)
    i_xy_past = corr_to_mi(r_xy_past)
    i_xy_future = corr_to_mi(r_xy_future)
    i_xp_yf = corr_to_mi(r_xp_yf)
    i_yp_xf = corr_to_mi(r_yp_xf)

    # 5. 计算冗余（MMI近似：取最小值）
    # Red->Red: 时间持久冗余
    # 当前时刻的冗余
    red_past = min(i_xx * (r_xy_past**2), i_yy * (r_xy_past**2))
    # 未来时刻的冗余
    red_future = min(i_xx * (r_xy_future**2), i_yy * (r_xy_future**2))
    # 时间持久冗余：加权平均
    red_red = (red_past + red_future) / 2

    # 6. 计算协同（改进版）
    # Syn->Syn: 时间持久协同
    # 跨时间的协同信息
    cross_mi = (i_xp_yf + i_yp_xf) / 2

    # 协同 = 未来互信息 - 预期的冗余部分
    # 如果跨时间信息 > 预期的冗余，则存在协同
    syn_syn = np.maximum(0, i_xy_future - red_future - cross_mi * 0.5)

    return float(syn_syn), float(red_red)


# ============================================================
# 完整 PhiID 计算（与论文一致）
# ============================================================
def compute_single_pair(args):
    """
    使用完整 PhiID 计算单个头对的协同/冗余

    Args:
        args: (ts1, ts2) 元组

    Returns:
        (syn_syn, red_red) 元组
    """
    ts1, ts2 = args

    try:
        if USE_FAST_APPROXIMATION:
            return fast_correlation_syn_red(ts1, ts2, tau=1, n_samples=50)

        # 完整的 PhiID 计算（与论文一致）
        atoms_res, _ = calc_PhiID(ts1, ts2, tau=1, kind="gaussian", redundancy="MMI")

        # 对时间序列求平均（论文方法）
        syn_syn = float(np.nanmean(np.asarray(atoms_res["sts"])))
        red_red = float(np.nanmean(np.asarray(atoms_res["rtr"])))

        return syn_syn, red_red
    except Exception:
        return np.nan, np.nan


def compute_single_pair_wrapper(ts1, ts2, pair_idx):
    """
    多进程包装函数

    Args:
        ts1: 时间序列1
        ts2: 时间序列2
        pair_idx: 头对索引

    Returns:
        (pair_idx, syn, red) 元组
    """
    syn, red = compute_single_pair((ts1, ts2))
    return (pair_idx, syn, red)


def main():
    csv_path = "./results/Gemma3-4B-Instruct/atten_activation_L2.csv"
    if not os.path.exists(csv_path):
        print(f"❌ 找不到数据文件 {csv_path}。请先运行: python src/activation_collection.py")
        return

    print("📊 正在加载 CSV 数据...")
    df = pd.read_csv(csv_path)

    df['Head_UID'] = df['Layer'].astype(str) + "_" + df['Head'].astype(str)
    unique_heads = df['Head_UID'].unique()
    all_prompt_ids = df['Prompt_ID'].unique()

    # 固定随机种子
    np.random.seed(42)

    # 采样 Prompts
    if PROMPT_SAMPLE_FRACTION < 1.0:
        num_prompts_to_sample = max(1, int(len(all_prompt_ids) * PROMPT_SAMPLE_FRACTION))
        prompt_ids = np.random.choice(all_prompt_ids, num_prompts_to_sample, replace=False)
        print(f"📈 采样 {num_prompts_to_sample}/{len(all_prompt_ids)} 个 Prompts")
    else:
        prompt_ids = all_prompt_ids

    print(f"✅ 数据加载完成！共 {len(unique_heads)} 个注意力头，{len(prompt_ids)} 个 Prompts。")

    # 生成头对索引
    num_heads = len(unique_heads)
    head_pairs_idx = list(combinations(range(num_heads), 2))
    num_pairs = len(head_pairs_idx)
    head_pairs_list = list(combinations(unique_heads, 2))
    print(f"🔗 共需计算 {num_pairs} 对头组合。")

    # 预估
    total_calculations = num_pairs * len(prompt_ids)
    print(f"\n⏱️  预估计算量: {total_calculations:,} 次")

    # 初始化累加器
    pair_accumulators = {
        idx: {'syn_sum': 0.0, 'red_sum': 0.0, 'count': 0}
        for idx in range(num_pairs)
    }

    # 初始化头对详细数据存储（用于保存每个头对的syn和red值）
    pair_detailed_records = []

    step_cols = [f'Step_{i+1}' for i in range(100)]

    # 开始计算
    print(f"\n🚀 启动 {N_WORKERS} 进程并行计算...")

    # 为每个 prompt 预处理数据
    for pid in prompt_ids:
        prompt_df = df[df['Prompt_ID'] == pid].set_index('Head_UID')

        # 预提取时间序列
        ts_list = []
        for h in unique_heads:
            ts_list.append(prompt_df.loc[h, step_cols].values.astype(np.float64))

        # 生成所有头对计算任务
        tasks = [(ts_list[i1], ts_list[i2], pair_idx)
                 for pair_idx, (i1, i2) in enumerate(head_pairs_idx)]

        # 多进程计算
        with Pool(N_WORKERS) as pool:
            # 使用 starmap 一次性提交所有任务
            results = pool.starmap(compute_single_pair_wrapper, tasks)

        # 聚合结果并保存详细记录
        for pair_idx, syn, red in results:
            if not np.isnan(syn) and not np.isnan(red):
                pair_accumulators[pair_idx]['syn_sum'] += syn
                pair_accumulators[pair_idx]['red_sum'] += red
                pair_accumulators[pair_idx]['count'] += 1

                # 保存每个头对的详细记录
                h1, h2 = head_pairs_list[pair_idx]
                layer1, head1 = map(int, h1.split('_'))
                layer2, head2 = map(int, h2.split('_'))
                pair_detailed_records.append({
                    'Prompt_ID': pid,
                    'Head1_UID': h1,
                    'Head2_UID': h2,
                    'Layer1': layer1,
                    'Head1': head1,
                    'Layer2': layer2,
                    'Head2': head2,
                    'Syn': syn,
                    'Red': red
                })

    print(f"✅ 计算完成！")

    # ---------------------------------------------------------
    # 数据聚合与计算 Synergy-Redundancy Rank
    # ---------------------------------------------------------
    print("\n🧮 正在聚合平均 Synergy 和 Redundancy...")

    # 计算每个 pair 的平均值
    avg_pair_results = {}
    for idx in range(num_pairs):
        vals = pair_accumulators[idx]
        count = vals['count']
        if count > 0:
            avg_pair_results[head_pairs_list[idx]] = {
                'avg_syn': vals['syn_sum'] / count,
                'avg_red': vals['red_sum'] / count
            }

    # 聚合到单个 Head
    head_metrics = {h: {'total_syn': 0.0, 'total_red': 0.0, 'count': 0} for h in unique_heads}

    for (h1, h2), metrics in avg_pair_results.items():
        for h in (h1, h2):
            head_metrics[h]['total_syn'] += metrics['avg_syn']
            head_metrics[h]['total_red'] += metrics['avg_red']
            head_metrics[h]['count'] += 1

    # 生成最终的 Head 级别指标
    final_head_data = []
    for h, data in head_metrics.items():
        if data['count'] > 0:
            layer, head = map(int, h.split('_'))
            final_head_data.append({
                'Layer': layer,
                'Head': head,
                'Syn': data['total_syn'] / data['count'],
                'Red': data['total_red'] / data['count']
            })

    result_df = pd.DataFrame(final_head_data)

    # 计算 Rank
    result_df['Syn_Rank'] = result_df['Syn'].rank(method='dense')
    result_df['Red_Rank'] = result_df['Red'].rank(method='dense')
    result_df['Syn_Red_Rank'] = result_df['Syn_Rank'] - result_df['Red_Rank']

    # 保存结果
    result_df = result_df.sort_values(by=['Layer', 'Head'])
    output_path = "./results/Gemma3-4B-Instruct/head_syn_red_ranks.csv"
    result_df.to_csv(output_path, index=False)

    print(f"✅ 头级别指标已保存至 {output_path}")
    print(f"   共处理 {len(final_head_data)} 个注意力头")

    # ---------------------------------------------------------
    # 保存头对详细数据（用于构建网络图）
    # ---------------------------------------------------------
    print("\n💾 正在保存头对详细数据...")

    # 创建头对数据DataFrame
    pair_df = pd.DataFrame(pair_detailed_records)

    # 计算每个头对的平均Syn和Red（跨所有prompts）
    pair_avg_df = pair_df.groupby(['Head1_UID', 'Head2_UID', 'Layer1', 'Head1', 'Layer2', 'Head2']).agg({
        'Syn': 'mean',
        'Red': 'mean'
    }).reset_index()

    # 重命名列以表示平均值
    pair_avg_df = pair_avg_df.rename(columns={'Syn': 'Avg_Syn', 'Red': 'Avg_Red'})

    # 保存详细数据（每个prompt的每个头对）
    pair_output_path = "./results/Gemma3-4B-Instruct/atten_pair_syn_red.csv"
    pair_df.to_csv(pair_output_path, index=False)

    # 保存平均数据（用于网络分析）
    pair_avg_output_path = "./results/Gemma3-4B-Instruct/atten_pair_syn_red_avg.csv"
    pair_avg_df.to_csv(pair_avg_output_path, index=False)

    print(f"✅ 头对详细数据已保存至 {pair_output_path}")
    print(f"   共 {len(pair_df)} 条记录（{len(prompt_ids)} prompts × {num_pairs} 头对）")
    print(f"✅ 头对平均数据已保存至 {pair_avg_output_path}")
    print(f"   共 {len(pair_avg_df)} 条记录（{num_pairs} 头对的平均值）")


if __name__ == "__main__":
    main()
