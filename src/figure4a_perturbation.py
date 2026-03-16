"""
复现 Figure 4a: 行为分歧曲线

扰动注意力头并计算KL散度，评估协同核心的重要性

KL散度定义（Teacher Forcing方式）:
    D_KL(P || Q) = (1/T) * Σ_{t=1}^{T} KL(P(w_t | S_{<t}) || Q(w_t | S_{<t}))

其中：
    - P(w_t | S_{<t}) 是原始模型在给定前缀 S_{<t} 时对位置t的词表分布预测
    - Q(w_t | S_{<t}) 是扰动模型在相同前缀下的词表分布预测
    - T 是评估序列的长度

方法：
    - 不让模型自由生成，而是使用固定前缀（Teacher Forcing）
    - 对每个位置计算完整词表分布的KL散度
    - 对所有位置求平均
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

# 设置输出目录
output_dir = "./results/Gemma3-4B-Instruct"
os.makedirs(output_dir, exist_ok=True)

# 设置绘图风格
sns.set_theme(style="white")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'sans-serif']


# 6个prompts（与activation_collection.py中一致，来自论文Table 2 & Table 3）
PROMPTS = [
    "Correct the error: He go to school every day.",
    "Identify the parts of speech in the sentence: Quickly, the agile cat climbed the tall tree.",
    "If you have 15 apples and you give away 5, how many do you have left?",
    "If it starts raining while the sun is shining, what weather phenomenon might you expect to see?",
    "Imagine a future where humans have evolved to live underwater. Describe the adaptations they might develop.",
    "Write a dialogue between two characters where one comforts the other after a loss, demonstrating empathy."
]

# 模型路径（使用本地已下载的权重）
MODEL_PATH = "/data/zjj/Synergistic_Core/Gemma-3-4B-Instruct"


def load_model_and_tokenizer():
    """加载模型和tokenizer"""
    print(f"加载模型: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,  # 【关键修改】Gemma3 必须使用 bfloat16 防溢出
        device_map="auto"
    )
    model.eval()
    print("✅ 模型加载完成")
    return model, tokenizer


def get_head_syn_red_rank():
    """获取注意力头的Syn-Red Rank排序"""
    df = pd.read_csv(f"{output_dir}/head_syn_red_ranks.csv")

    # 先检查原始数据的统计信息
    print(f"\n📊 原始Syn-Red Rank统计:")
    print(f"   最大值: {df['Syn_Red_Rank'].max():.2f}")
    print(f"   最小值: {df['Syn_Red_Rank'].min():.2f}")
    print(f"   平均值: {df['Syn_Red_Rank'].mean():.2f}")
    print(f"   中位数: {df['Syn_Red_Rank'].median():.2f}")
    print(f"   正值数量: {(df['Syn_Red_Rank'] > 0).sum()}")
    print(f"   负值数量: {(df['Syn_Red_Rank'] < 0).sum()}")

    # 关键问题：论文中高Syn-Red Rank = 协同核心（高Syn，低Red）
    # 但如果大部分值是负的，说明排序方向需要调整

    # 让我们先尝试从低到高排序（负值在前面 = 更协同）
    df_sorted_asc = df.sort_values(by='Syn_Red_Rank', ascending=True)
    df_sorted_desc = df.sort_values(by='Syn_Red_Rank', ascending=False)

    # 检查两种排序的前10个头
    print(f"\n📊 升序排序（从低到高）前5个头:")
    for i in range(min(5, len(df_sorted_asc))):
        row = df_sorted_asc.iloc[i]
        uid = f"{row['Layer']}_{row['Head']}"
        print(f"   {i+1}. {uid}: Syn_Red_Rank={row['Syn_Red_Rank']:.2f}, Syn={row['Syn']:.4f}, Red={row['Red']:.4f}")

    print(f"\n📊 降序排序（从高到低）前5个头:")
    for i in range(min(5, len(df_sorted_desc))):
        row = df_sorted_desc.iloc[i]
        uid = f"{row['Layer']}_{row['Head']}"
        print(f"   {i+1}. {uid}: Syn_Red_Rank={row['Syn_Red_Rank']:.2f}, Syn={row['Syn']:.4f}, Red={row['Red']:.4f}")

    # 按Syn_Red_Rank从高到低排序（默认）
    df_sorted = df_sorted_desc

    # 创建头UID列表
    head_uids = df_sorted['Layer'].astype(str) + '_' + df_sorted['Head'].astype(str)
    syn_red_ranks = df_sorted['Syn_Red_Rank'].values

    print(f"\n✅ 加载了 {len(head_uids)} 个注意力头的Syn-Red Rank")
    return list(head_uids), syn_red_ranks


def get_gemma3_layer(model, layer_idx):
    """
    准确获取 Gemma 3 的语言模型层 (适配多模态嵌套架构)
    """
    # 按照你打印出来的结构，Gemma 3 的文本层在这里：
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        return model.model.language_model.layers[layer_idx]
    # 兼容备用方案
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers[layer_idx]
    elif hasattr(model, 'layers'):
        return model.layers[layer_idx]
    else:
        raise AttributeError(f"无法在模型中找到 layer_idx={layer_idx} 的层结构！")

def deactivate_head(model, layer_idx, head_idx):
    """
    【核心修复】真正的消融实验：通过将 o_proj 中该 Head 的输出权重置零，
    彻底阻断其向残差流(Residual Stream)写入信息。
    仅保存对应的切片，避免同层多次修改时互相覆盖。
    """
    layer = get_gemma3_layer(model, layer_idx)
    o_proj = layer.self_attn.o_proj
    
    # 动态获取总头数
    if hasattr(model.config, 'text_config'):
        num_heads = model.config.text_config.num_attention_heads
    else:
        num_heads = model.config.num_attention_heads

    # 计算 head_dim
    head_dim = layer.self_attn.q_proj.weight.shape[0] // num_heads

    # 计算该 Head 在 o_proj 中的起始和结束列索引
    # o_proj shape: [hidden_size, num_heads * head_dim]
    start_idx = head_idx * head_dim
    end_idx = (head_idx + 1) * head_dim

    # 1. 仅克隆这一个 Head 的切片保存
    o_slice_orig = o_proj.weight.data[:, start_idx:end_idx].clone()

    # 2. 将该 Head 的权重彻底置零 (Deactivation/Ablation)
    o_proj.weight.data[:, start_idx:end_idx] = 0.0

    return {
        'layer_idx': layer_idx,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'o_slice': o_slice_orig
    }

def restore_head_parameters(model, original_params):
    """
    【核心修复】仅将之前置零的切片精准恢复
    """
    layer_idx = original_params['layer_idx']
    start_idx = original_params['start_idx']
    end_idx = original_params['end_idx']
    
    layer = get_gemma3_layer(model, layer_idx)
    device = layer.self_attn.o_proj.weight.device
    
    # 将保存的切片原封不动地塞回去
    layer.self_attn.o_proj.weight.data[:, start_idx:end_idx] = original_params['o_slice'].to(device)

def get_generated_sequence(model, tokenizer, prompt, max_length=20):
    """
    使用原始模型生成完整序列

    Args:
        model: 模型
        tokenizer: tokenizer
        prompt: 输入prompt
        max_length: 最大生成token数

    Returns:
        full_sequence: 完整序列（包括prompt和生成的tokens）
        generated_ids: 生成的token IDs
        input_length: prompt长度
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    full_sequence = outputs[0]  # 完整序列
    generated_ids = full_sequence[input_length:]  # 只取生成的部分

    return full_sequence, generated_ids, input_length


def get_vocabulary_distribution_at_position(model, tokenizer, input_ids, position, past_key_values=None):
    """
    【修正版】正确处理 past_key_values，防止位置编码错乱导致 NaN
    """
    # 核心修复：如果有历史缓存，仅输入当前 1 个 Token；如果没有，输入前面的所有 Token
    if past_key_values is None:
        context_ids = input_ids[:position].unsqueeze(0)  # [1, position]
    else:
        context_ids = input_ids[position-1:position].unsqueeze(0)  # [1, 1]

    with torch.no_grad():
        outputs = model(
            input_ids=context_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=False,
            output_attentions=False
        )

        # 取最后一个预测的 logits
        logits = outputs.logits[0, -1, :] 

        # 转换为 float32 计算 log_softmax，确保数值极度稳定
        logits = logits.float()
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        return log_probs.cpu().numpy(), outputs.past_key_values


def compute_average_kl_divergence(log_probs_p_list, log_probs_q_list, epsilon=1e-10):
    """
    计算两个分布序列的平均KL散度

    定义：D_{KL}(P || Q) = (1/T) * Σ_{t=1}^{T} KL(P(w_t | S_{<t}) || Q(w_t | S_{<t}))

    其中：
    - P(w_t | S_{<t}) 是原始模型在位置t的词表分布
    - Q(w_t | S_{<t}) 是扰动模型在位置t的词表分布
    - T 是评估序列的长度

    Args:
        log_probs_p_list: 原始模型的词表对数概率分布列表
        log_probs_q_list: 扰动模型的词表对数概率分布列表
        epsilon: 防止数值不稳定的小常数

    Returns:
        avg_kl_div: 平均KL散度，如果计算失败返回nan
    """
    # 检查长度是否一致
    if len(log_probs_p_list) != len(log_probs_q_list):
        print(f"    警告: 序列长度不匹配 ({len(log_probs_p_list)} vs {len(log_probs_q_list)})")
        return np.nan

    if len(log_probs_p_list) == 0:
        print(f"    警告: 空的log_probs列表")
        return np.nan

    try:
        # 对每个位置计算KL散度（数值稳定版本）
        kl_divs = []
        for t in range(len(log_probs_p_list)):
            # 获取位置t的词表分布
            log_probs_p = log_probs_p_list[t].astype(np.float64)  # [vocab_size]
            log_probs_q = log_probs_q_list[t].astype(np.float64)  # [vocab_size]

            # 检查是否包含nan
            if np.any(np.isnan(log_probs_p)) or np.any(np.isnan(log_probs_q)):
                print(f"    警告: 位置{t}的log_probs包含nan")
                continue

            # 数值稳定的KL计算
            # KL(P||Q) = sum(P * (log_P - log_Q))
            # 其中 P = exp(log_P - logsumexp(log_P))

            # 计算 logsumexp(log_P) 用于归一化
            log_sum_p = np.logaddexp.reduce(log_probs_p)

            # 归一化 log_P（减去 log_sum 等价于除以 sum(exp(log_P))）
            log_probs_p_norm = log_probs_p - log_sum_p

            # 计算 P = exp(log_P_norm)，这个操作是安全的因为 log_P_norm <= 0
            probs_p = np.exp(log_probs_p_norm)

            # 计算 log_Q - log_P (在概率空间)
            log_ratio = log_probs_q - log_probs_p

            # KL = sum(P * (log_P - log_Q)) = -sum(P * log_ratio)
            kl_t = -np.sum(probs_p * log_ratio)

            # 检查结果有效性
            if np.isnan(kl_t) or np.isinf(kl_t):
                print(f"    警告: 位置{t}的KL值为{kl_t}，跳过")
                continue

            kl_divs.append(kl_t)

        if len(kl_divs) == 0:
            print(f"    警告: 所有位置的KL计算都失败了")
            return np.nan

        # 返回平均KL散度
        avg_kl_div = np.mean(kl_divs)
        return avg_kl_div

    except Exception as e:
        print(f"    警告: KL计算失败 - {e}")
        import traceback
        traceback.print_exc()
        return np.nan


def perturb_heads_and_compute_kl(model, tokenizer, head_uids_to_perturb, noise_std=0.1):
    """
    扰动指定的注意力头并计算KL散度

    论文方法：
    1. 原始模型生成完整序列
    2. 消融模型基于相同的生成历史预测每一步
    3. 计算：D_KL(P || Q) = (1/T) * Σ KL(P(w_t | x^(na)_<t) || Q(w_t | x^(na)_<t))

    其中 x^(na)_<t 是原始模型生成的序列

    Args:
        model: 模型
        tokenizer: tokenizer
        head_uids_to_perturb: 要扰动的头UID列表
        noise_std: 噪声标准差

    Returns:
        avg_kl_div: 平均KL散度（跨所有prompts）
    """
    all_kl_divs = []

    for prompt_idx, prompt in enumerate(PROMPTS):
        try:
            # 步骤1: 使用原始模型生成完整序列
            full_sequence, generated_ids, input_length = get_generated_sequence(
                model, tokenizer, prompt, max_length=20
            )

            if len(generated_ids) == 0:
                print(f"    Prompt {prompt_idx}: 生成序列为空")
                continue

            # 步骤2: 对每个生成位置，获取原始模型的词表分布 P(w_t | x^(na)_<t)
            log_probs_p_list = []
            past_key_values = None

            for t in range(len(generated_ids)):
                position = input_length + t  # 当前位置在完整序列中的索引
                log_probs, past_key_values = get_vocabulary_distribution_at_position(
                    model, tokenizer, full_sequence, position, past_key_values
                )
                log_probs_p_list.append(log_probs)

            # 步骤3: 扰动注意力头
            original_params_list = []
            for head_uid in head_uids_to_perturb:
                layer_idx, head_idx = map(int, head_uid.split('_'))
                # 修改后：
                original_params = deactivate_head(model, layer_idx, head_idx)
                original_params_list.append(original_params)

            # 步骤4: 对每个生成位置，获取消融模型的词表分布 Q(w_t | x^(na)_<t)
            # 关键：使用相同的生成历史 full_sequence
            log_probs_q_list = []
            past_key_values = None

            for t in range(len(generated_ids)):
                position = input_length + t
                log_probs, past_key_values = get_vocabulary_distribution_at_position(
                    model, tokenizer, full_sequence, position, past_key_values
                )
                log_probs_q_list.append(log_probs)

            # 步骤5: 恢复原始参数
            for original_params in original_params_list:
                restore_head_parameters(model, original_params)

            # 步骤6: 计算平均KL散度
            kl_div = compute_average_kl_divergence(log_probs_p_list, log_probs_q_list)

            if not np.isnan(kl_div):
                all_kl_divs.append(kl_div)
            else:
                print(f"    Prompt {prompt_idx}: KL为nan，跳过")

        except Exception as e:
            print(f"    Prompt {prompt_idx}: 处理失败 - {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(all_kl_divs) == 0:
        print(f"    所有prompt都失败了，返回nan")
        return np.nan

    avg_kl_div = np.mean(all_kl_divs)
    return avg_kl_div


def run_perturbation_experiment(model, tokenizer, head_uids, syn_ordered_head_uids,
                                 fractions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                 n_random_runs=5, noise_std=0.1):
    """
    运行扰动实验

    Args:
        model: 模型
        tokenizer: tokenizer
        head_uids: 所有头UID列表
        syn_ordered_head_uids: 按Syn-Red Rank排序的头UID列表
        fractions: 扰动比例列表
        n_random_runs: 随机运行次数
        noise_std: 噪声标准差

    Returns:
        results: 实验结果字典
    """
    n_heads = len(head_uids)

    results = {
        'fraction': [],
        'syn_ordered_kl': [],
        'random_ordered_kl_mean': [],
        'random_ordered_kl_std': []
    }

    print(f"\n{'=' * 60}")
    print(f"运行扰动实验")
    print(f"总头数: {n_heads}")
    print(f"扰动比例: {fractions}")
    print(f"随机运行次数: {n_random_runs}")
    print(f"噪声标准差: {noise_std}")
    print(f"{'=' * 60}")

    for fraction in fractions:
        n_perturb = int(n_heads * fraction)
        print(f"\n--- 扰动 {fraction*100:.0f}% 的头 ({n_perturb}/{n_heads}) ---")

        # 方式1: 按Syn-Red Rank顺序
        print(f"  计算Syn-Red顺序的KL散度...")
        syn_heads_to_perturb = syn_ordered_head_uids[:n_perturb]
        syn_kl = perturb_heads_and_compute_kl(model, tokenizer, syn_heads_to_perturb, noise_std)

        # 方式2: 随机顺序（运行多次）
        print(f"  计算随机顺序的KL散度（{n_random_runs}次运行）...")
        random_kls = []
        for run_idx in range(n_random_runs):
            np.random.seed(run_idx)  # 确保可重复性
            random_heads = np.random.choice(head_uids, size=n_perturb, replace=False)
            random_kl = perturb_heads_and_compute_kl(model, tokenizer, random_heads, noise_std)
            random_kls.append(random_kl)
            print(f"    运行 {run_idx+1}/{n_random_runs}: KL = {random_kl:.6f}")

        random_kl_mean = np.mean(random_kls)
        random_kl_std = np.std(random_kls)

        # 保存结果
        results['fraction'].append(fraction)
        results['syn_ordered_kl'].append(syn_kl)
        results['random_ordered_kl_mean'].append(random_kl_mean)
        results['random_ordered_kl_std'].append(random_kl_std)

        print(f"  Syn-Red顺序: KL = {syn_kl:.6f}")
        print(f"  随机顺序: KL = {random_kl_mean:.6f} ± {random_kl_std:.6f}")

    return results


def plot_figure4a(results):
    """
    绘制Figure 4a: 行为分歧曲线

    Args:
        results: 实验结果字典
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    fractions = results['fraction']
    syn_kl = results['syn_ordered_kl']
    random_kl_mean = results['random_ordered_kl_mean']
    random_kl_std = results['random_ordered_kl_std']

    # 绘制Syn-Red顺序曲线（实线）
    ax.plot(fractions, syn_kl, 'o-', color='#d62728', linewidth=2.5,
            markersize=8, label='Syn-Red order')

    # 绘制随机顺序曲线（虚线）
    ax.plot(fractions, random_kl_mean, 's--', color='#1f77b4', linewidth=2.5,
            markersize=8, label='Random order')

    # 添加随机顺序的标准差阴影
    ax.fill_between(fractions,
                    np.array(random_kl_mean) - np.array(random_kl_std),
                    np.array(random_kl_mean) + np.array(random_kl_std),
                    color='#1f77b4', alpha=0.2)

    ax.set_xlabel('Fraction of attention heads perturbed', fontsize=14, fontweight='bold')
    ax.set_ylabel('KL divergence (nats)', fontsize=14, fontweight='bold')
    ax.set_title('Behaviour Divergence: Perturbation Analysis (Gemma3-4B)',
                 fontsize=15, fontweight='bold', pad=15)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "figure4a_perturbation_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Figure 4a已保存至: {output_path}")
    plt.close()

    # 保存数据到CSV
    results_df = pd.DataFrame({
        'Fraction': fractions,
        'Syn_Red_Order_KL': syn_kl,
        'Random_Order_KL_Mean': random_kl_mean,
        'Random_Order_KL_Std': random_kl_std
    })
    csv_path = os.path.join(output_dir, "figure4a_perturbation_data.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"✅ 数据已保存至: {csv_path}")


def main():
    print("=" * 60)
    print("复现 Figure 4a: 行为分歧曲线")
    print("=" * 60)

    # 加载模型和tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # 获取Syn-Red Rank排序
    all_head_uids, syn_red_ranks = get_head_syn_red_rank()

    # 验证排序是否正确（打印前10个头）
    print(f"\n📊 验证Syn-Red Rank排序（前10个头）:")
    for i in range(min(10, len(all_head_uids))):
        print(f"   {i+1}. {all_head_uids[i]}: Syn_Red_Rank = {syn_red_ranks[i]:.2f}")

    print(f"\n📊 最高和最低的Syn-Red Rank:")
    print(f"   最高: {syn_red_ranks[0]:.2f} ({all_head_uids[0]})")
    print(f"   最低: {syn_red_ranks[-1]:.2f} ({all_head_uids[-1]})")

    syn_ordered_head_uids = [all_head_uids[i] for i in range(len(all_head_uids))]

    # 运行扰动实验
    results = run_perturbation_experiment(
        model, tokenizer,
        all_head_uids, syn_ordered_head_uids,
        fractions=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
        n_random_runs=5,
        noise_std=0.1
    )

    # 绘制Figure 4a
    print("\n" + "=" * 60)
    print("生成Figure 4a可视化")
    print("=" * 60)
    plot_figure4a(results)

    print("\n" + "=" * 60)
    print("✅ 分析完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
