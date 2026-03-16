"""
复现 Figure 4b: GSM8K Benchmark 准确率对比

扰动协同核心、冗余核心或随机子集，比较GSM8K数据集上的准确率

方法：
- 对选定的注意力头的 q_proj 和 o_proj 注入高斯噪声
- 在GSM8K测试集上评估准确率（自动缓存到本地）
- 对比三种扰动方式：协同核心、冗余核心、随机
"""

import os
import torch
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

# 设置输出目录
output_dir = "./results/Gemma3-4B-Instruct"
os.makedirs(output_dir, exist_ok=True)

# 模型路径
MODEL_PATH = "/data/zjj/Synergistic_Core/Gemma-3-4B-Instruct"

# GSM8K 数据集缓存路径
GSM8K_CACHE_DIR = "/data/zjj/Synergistic_Core/data/gsm8k"


def load_model_and_tokenizer():
    """
    加载HuggingFace模型和tokenizer

    Returns:
        model: HuggingFace模型
        tokenizer: tokenizer
        is_vllm: 始终返回False
    """
    print(f"加载模型: {MODEL_PATH}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    print("✅ HuggingFace模型加载完成")
    return model, tokenizer, False


def get_gemma3_layer(model, layer_idx):
    """获取 Gemma 3 的语言模型层"""
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        return model.model.language_model.layers[layer_idx]
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers[layer_idx]
    elif hasattr(model, 'layers'):
        return model.layers[layer_idx]
    else:
        raise AttributeError(f"无法找到 layer_idx={layer_idx} 的层")


def add_gaussian_noise_to_head(model, layer_idx, head_idx, noise_fraction=0.1):
    """向指定注意力头注入高斯噪声"""
    layer = get_gemma3_layer(model, layer_idx)

    q_proj = layer.self_attn.q_proj
    o_proj = layer.self_attn.o_proj

    device = q_proj.weight.device
    dtype = q_proj.weight.dtype

    # 获取头数和维度
    if hasattr(model.config, 'text_config'):
        num_heads = model.config.text_config.num_attention_heads
    else:
        num_heads = model.config.num_attention_heads

    head_dim = q_proj.weight.shape[0] // num_heads

    # 保存原始参数
    q_weight_orig = q_proj.weight.data.clone()
    o_weight_orig = o_proj.weight.data.clone()

    start_idx = head_idx * head_dim
    end_idx = (head_idx + 1) * head_dim

    # 获取当前切片
    q_slice = q_proj.weight.data[start_idx:end_idx, :]
    o_slice = o_proj.weight.data[:, start_idx:end_idx]
    
    # 【核心修改】计算当前切片的标准差，生成等比例噪声
    q_std = q_slice.std().item()
    o_std = o_slice.std().item()
    
    q_noise = torch.randn_like(q_slice) * (q_std * noise_fraction)
    o_noise = torch.randn_like(o_slice) * (o_std * noise_fraction)

    q_proj.weight.data[start_idx:end_idx, :] += q_noise
    o_proj.weight.data[:, start_idx:end_idx] += o_noise

    return {
        'layer_idx': layer_idx,
        'q_weight': q_weight_orig,
        'o_weight': o_weight_orig
    }


def restore_head_parameters(model, original_params):
    """恢复原始参数"""
    layer_idx = original_params['layer_idx']
    layer = get_gemma3_layer(model, layer_idx)

    layer.self_attn.q_proj.weight.data = original_params['q_weight'].to(layer.self_attn.q_proj.weight.device)
    layer.self_attn.o_proj.weight.data = original_params['o_weight'].to(layer.self_attn.o_proj.weight.device)


def load_gsm8k_test_samples(num_samples=100):
    """
    加载GSM8K测试集样本（从本地parquet文件）

    Args:
        num_samples: 加载的样本数量（默认100，用于快速测试）

    Returns:
        samples: 问题列表
    """
    import pandas as pd

    print(f"📊 加载GSM8K测试集")

    # 尝试从本地parquet文件加载
    test_parquet = os.path.join(GSM8K_CACHE_DIR, "main/test-00000-of-00001.parquet")

    if os.path.exists(test_parquet):
        print(f"   从本地parquet加载: {test_parquet}")

        # 读取parquet文件
        df = pd.read_parquet(test_parquet)
        print(f"   总样本数: {len(df)}")

        # 采样（固定种子确保可重复）
        np.random.seed(42)
        if len(df) > num_samples:
            indices = np.random.choice(len(df), num_samples, replace=False)
            df = df.iloc[indices].reset_index(drop=True)

        # 转换为统一格式
        formatted_samples = []
        for _, row in df.iterrows():
            question = row['question']
            answer = row['answer']

            # 提取最终答案（#### 后面的数字）
            final_answer = None
            if '####' in answer:
                final_answer = answer.split('####')[-1].strip()

            formatted_samples.append({
                'question': question,
                'answer': answer,
                'final_answer': final_answer
            })

        print(f"✅ 加载了 {len(formatted_samples)} 个GSM8K测试样本")
        return formatted_samples

    # 备用方案：从HuggingFace加载
    print(f"   本地文件不存在，尝试从HuggingFace加载...")
    from datasets import load_dataset

    # 尝试多个GSM8K数据集名称
    dataset_names = [
        "gsm8k",
        "openai/gsm8k",
        "allenai/gsm8k",
    ]

    dataset = None
    for name in dataset_names:
        try:
            print(f"   尝试加载数据集: {name}")
            dataset = load_dataset(
                name,
                split="test",
                cache_dir=GSM8K_CACHE_DIR,
                download_mode="reuse_cache_if_exists"
            )
            print(f"   ✅ 成功加载GSM8K测试集")
            break
        except Exception as e:
            print(f"   ❌ 加载失败: {e}")
            continue

    if dataset is None:
        raise FileNotFoundError(
            f"无法加载GSM8K数据集。\n"
            f"本地文件: {test_parquet}\n"
            f"HuggingFace加载失败。"
        )

    print(f"   总共 {len(dataset)} 个测试样本")

    # 采样（固定种子确保可重复）
    np.random.seed(42)
    if len(dataset) > num_samples:
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        samples = [dataset[i] for i in indices]
    else:
        samples = list(dataset)

    print(f"✅ 加载了 {len(samples)} 个GSM8K测试样本")

    # 转换为统一格式
    formatted_samples = []
    for sample in samples:
        # GSM8K格式: {'question': ..., 'answer': '... #### <数字>'}
        question = sample['question']
        answer = sample['answer']

        # 提取最终答案（#### 后面的数字）
        final_answer = None
        if '####' in answer:
            final_answer = answer.split('####')[-1].strip()

        formatted_samples.append({
            'question': question,
            'answer': answer,
            'final_answer': final_answer
        })

    return formatted_samples


def format_gsm8k_prompt(question):
    """格式化GSM8K问题为prompt"""
    # 使用Few-Shot格式
    prompt = f"Question: {question}\nAnswer:"
    return prompt


def check_gsm8k_answer_correctness(generated_text, ground_truth, final_answer=None):
    """
    检查GSM8K答案是否正确

    Args:
        generated_text: 模型生成的文本
        ground_truth: 标准完整答案（包含推理过程）
        final_answer: 提取的最终答案（数字）

    Returns:
        correct: 是否正确
    """
    if final_answer is None:
        # 从ground_truth中提取最终答案
        if '####' in ground_truth:
            final_answer = ground_truth.split('####')[-1].strip()
        else:
            final_answer = ground_truth.strip()

    # 方法1: 检查生成文本中是否包含正确答案
    # 标准化答案格式（去除逗号、空格等）
    def normalize_number(num_str):
        num_str = str(num_str).strip().replace(',', '').replace(' ', '')
        # 处理分数
        if '\\' in num_str or '/' in num_str:
            try:
                if '\\frac' in num_str:
                    # LaTeX分数格式
                    import re
                    match = re.search(r'\\frac\{([^}]+)\}\{([^}]+)\}', num_str)
                    if match:
                        num = float(match.group(1)) / float(match.group(2))
                        return str(num)
                else:
                    # 简单分数
                    parts = num_str.split('/')
                    if len(parts) == 2:
                        num = float(parts[0]) / float(parts[1])
                        return str(num)
            except:
                pass
        return num_str

    final_answer_norm = normalize_number(final_answer)
    generated_text_norm = normalize_number(generated_text)

    # 检查是否包含正确答案
    if final_answer_norm in generated_text_norm:
        return True

    # 方法2: 尝试提取生成的最后数字
    import re
    # 查找所有数字（包括小数、分数）
    numbers_in_text = re.findall(r'\d+\.?\d*|\d+/\d+', generated_text.replace('\\', '/'))
    if numbers_in_text:
        last_number = numbers_in_text[-1]
        # 尝试转换和比较
        try:
            # 处理分数
            if '/' in last_number:
                parts = last_number.split('/')
                if len(parts) == 2:
                    gen_num = float(parts[0]) / float(parts[1])
                else:
                    gen_num = float(last_number)
            else:
                gen_num = float(last_number)

            # 处理标准答案
            if '/' in final_answer_norm:
                parts = final_answer_norm.split('/')
                if len(parts) == 2:
                    gt_num = float(parts[0]) / float(parts[1])
                else:
                    gt_num = float(final_answer_norm)
            else:
                gt_num = float(final_answer_norm)

            if abs(gen_num - gt_num) < 1e-6:
                return True
        except:
            pass

    return False


def evaluate_gsm8k_accuracy(model, tokenizer, samples, max_new_tokens=1024):
    """
    在GSM8K样本上评估准确率（HuggingFace模型）

    Args:
        model: HuggingFace模型
        tokenizer: tokenizer
        samples: GSM8K样本列表
        max_new_tokens: 最大生成token数

    Returns:
        accuracy: 准确率
        correct_count: 正确数量
        total_count: 总数量
    """
    correct_count = 0
    total_count = len(samples)

    print(f"\n   评估 {total_count} 个GSM8K问题...")

    for idx, sample in enumerate(tqdm(samples, desc="   评估进度")):
        question = sample["question"]
        ground_truth = sample["answer"]
        final_answer = sample.get("final_answer")

        # 格式化prompt
        prompt = format_gsm8k_prompt(question)

        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            # 解码生成的文本
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

            # 检查正确性
            if check_gsm8k_answer_correctness(generated_text, ground_truth, final_answer):
                correct_count += 1

        except Exception as e:
            print(f"\n   警告: 样本 {idx} 评估失败: {e}")
            continue

    accuracy = correct_count / total_count if total_count > 0 else 0.0

    print(f"\n   准确率: {accuracy:.4f} ({correct_count}/{total_count})")

    return accuracy, correct_count, total_count


def perturb_and_evaluate(model, tokenizer, samples, head_uids_to_perturb, noise_std=0.1):
    """
    扰动注意力头并评估GSM8K准确率

    Args:
        model: 模型
        tokenizer: tokenizer
        samples: GSM8K样本
        head_uids_to_perturb: 要扰动的头UID列表
        noise_std: 噪声标准差

    Returns:
        accuracy: 准确率
    """
    if len(head_uids_to_perturb) == 0:
        print("\n   无需扰动（0个头），直接评估...")
        return evaluate_gsm8k_accuracy(model, tokenizer, samples)

    print(f"\n   扰动 {len(head_uids_to_perturb)} 个注意力头...")

    # 扰动注意力头
    original_params_list = []
    for head_uid in head_uids_to_perturb:
        layer_idx, head_idx = map(int, head_uid.split('_'))
        original_params = add_gaussian_noise_to_head(model, layer_idx, head_idx, noise_std)
        original_params_list.append(original_params)

    # 评估准确率
    try:
        accuracy, correct_count, total_count = evaluate_gsm8k_accuracy(model, tokenizer, samples)
    finally:
        # 恢复原始参数
        for original_params in original_params_list:
            restore_head_parameters(model, original_params)
        print("   ✅ 参数已恢复")

    return accuracy


def run_gsm8k_accuracy_experiment(model, tokenizer, syn_core_heads, red_core_heads, all_head_uids,
                                   noise_fraction=0.1, num_samples=100):
    """
    运行GSM8K准确率实验（4组：baseline、syn、red、随机）

    Args:
        model: HuggingFace模型
        tokenizer: tokenizer
        syn_core_heads: 协同核心头UID列表（前25%）
        red_core_heads: 冗余核心头UID列表（后25%）
        all_head_uids: 所有头UID列表
        noise_fraction: 自适应噪声比例（相对于权重标准差）
        num_samples: GSM8K样本数量

    Returns:
        results: 实验结果字典
    """
    # 加载GSM8K测试集
    gsm8k_samples = load_gsm8k_test_samples(num_samples=num_samples)

    n_heads = len(all_head_uids)
    fraction = 0.25  # 固定25%扰动
    n_perturb = int(n_heads * fraction)

    results = {
        'baseline_acc': None,
        'syn_core_acc': None,
        'red_core_acc': None,
        'random_acc_mean': None,
        'random_acc_std': None,
        'random_accs': []
    }

    print(f"\n{'=' * 60}")
    print(f"GSM8K Benchmark 准确率实验")
    print(f"总头数: {n_heads}")
    print(f"扰动头数: {n_perturb} (25%)")
    print(f"协同核心: {len(syn_core_heads)} 个头 (前25%)")
    print(f"冗余核心: {len(red_core_heads)} 个头 (后25%)")
    print(f"GSM8K样本数: {len(gsm8k_samples)}")
    print(f"自适应噪声比例: {noise_fraction} (权重标准差的{noise_fraction*100}%)")
    print(f"{'=' * 60}")

    # 第1组: Baseline（完整模型，无扰动）
    print(f"\n1️⃣ Baseline（完整模型，无扰动）...")
    baseline_acc, _, _ = evaluate_gsm8k_accuracy(model, tokenizer, gsm8k_samples)
    results['baseline_acc'] = baseline_acc

    # 第2-4组: 扰动实验
    # 第2组: 扰动协同核心
    print(f"\n2️⃣ 扰动协同核心 (前{min(n_perturb, len(syn_core_heads))}个头)...")
    syn_heads_to_perturb = syn_core_heads[:n_perturb]
    print(f"   扰动头数: {len(syn_heads_to_perturb)}")
    syn_acc = perturb_and_evaluate(model, tokenizer, gsm8k_samples, syn_heads_to_perturb, noise_fraction)
    results['syn_core_acc'] = syn_acc

    # 第3组: 扰动冗余核心
    print(f"\n3️⃣ 扰动冗余核心 (前{min(n_perturb, len(red_core_heads))}个头)...")
    red_heads_to_perturb = red_core_heads[:n_perturb]
    print(f"   扰动头数: {len(red_heads_to_perturb)}")
    red_acc = perturb_and_evaluate(model, tokenizer, gsm8k_samples, red_heads_to_perturb, noise_fraction)
    results['red_core_acc'] = red_acc

    # 第4组: 随机扰动（单次运行）
    print(f"\n4️⃣ 随机扰动 (单次运行)...")
    np.random.seed(42)  # 固定种子确保可重复
    random_heads = np.random.choice(all_head_uids, size=n_perturb, replace=False)
    print(f"   扰动 {len(random_heads)} 个头")
    random_acc = perturb_and_evaluate(model, tokenizer, gsm8k_samples, random_heads, noise_fraction)
    results['random_acc_mean'] = random_acc
    results['random_acc_std'] = 0.0
    results['random_accs'] = [random_acc]

    print(f"\n📊 总结 (25% 扰动):")
    print(f"   Baseline: {baseline_acc:.4f}")
    print(f"   协同核心: {syn_acc:.4f} (下降: {baseline_acc - syn_acc:.4f})")
    print(f"   冗余核心: {red_acc:.4f} (下降: {baseline_acc - red_acc:.4f})")
    print(f"   随机: {random_acc:.4f} (下降: {baseline_acc - random_acc:.4f})")

    return results


def plot_figure4b(results):
    """
    绘制Figure 4b: GSM8K准确率对比图（4组：baseline、syn、red、随机）

    Args:
        results: 实验结果字典
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'sans-serif']

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    baseline_acc = results['baseline_acc']
    syn_acc = results['syn_core_acc']
    red_acc = results['red_core_acc']
    random_acc_mean = results['random_acc_mean']
    random_acc_std = results['random_acc_std']

    # 数据（4组）
    methods = ['Baseline\n(No Perturbation)', 'Synergistic\nCore (Top 25%)',
               'Redundant\nCore (Bottom 25%)', 'Random\n(25%)']
    accuracies = [baseline_acc, syn_acc, red_acc, random_acc_mean]
    errors = [0, 0, 0, random_acc_std]
    colors = ['#95a5a6', '#d62728', '#1f77b4', '#2ca02c']

    # 绘制条形图
    x_pos = np.arange(len(methods))
    bars = ax.bar(x_pos, accuracies, yerr=errors, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=1.5, capsize=10, error_kw={'linewidth': 2})

    # 添加数值标签
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        if i == 3:  # 随机有标准差
            label = f'{acc:.3f}±{random_acc_std:.3f}'
        else:
            label = f'{acc:.3f}'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))

    # 添加下降百分比标注
    for i in range(1, 4):  # 跳过baseline
        drop = baseline_acc - accuracies[i]
        drop_pct = (drop / baseline_acc) * 100 if baseline_acc > 0 else 0
        height = accuracies[i]
        ax.text(bar.get_x() + bar.get_width()/2., height / 2,
                f'-{drop_pct:.1f}%', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel('GSM8K Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('GSM8K Benchmark: Accuracy with 25% Head Perturbation (Gemma3-4B)',
                 fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(accuracies) * 1.2)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "figure4b_gsm8k_accuracy.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Figure 4b已保存至: {output_path}")
    plt.close()

    # 保存数据到CSV
    results_df = pd.DataFrame({
        'Method': ['Baseline', 'Synergistic_Core', 'Redundant_Core', 'Random'],
        'Accuracy': [baseline_acc, syn_acc, red_acc, random_acc_mean],
        'Std': [0, 0, 0, random_acc_std],
        'Drop_from_Baseline': [0, baseline_acc - syn_acc, baseline_acc - red_acc, baseline_acc - random_acc_mean]
    })
    csv_path = os.path.join(output_dir, "figure4b_gsm8k_accuracy.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"✅ 数据已保存至: {csv_path}")


def main():
    print("=" * 60)
    print("复现 Figure 4b: GSM8K Benchmark 准确率对比")
    print("=" * 60)

    # 加载HuggingFace模型（统一使用HF模型进行所有实验）
    print("\n📦 加载HuggingFace模型")
    model, tokenizer, _ = load_model_and_tokenizer()

    # 读取Syn-Red Rank数据
    print("\n" + "=" * 60)
    print("加载Syn-Red Rank数据")
    print("=" * 60)

    df = pd.read_csv(f"{output_dir}/head_syn_red_ranks.csv")

    # 直接按Syn_Red_Rank排序（不过滤）
    df_sorted = df.sort_values(by='Syn_Red_Rank', ascending=False)

    # 创建头UID列表
    all_head_uids = df_sorted['Layer'].astype(str) + '_' + df_sorted['Head'].astype(str)
    all_head_uids = list(all_head_uids)

    # 协同核心：Syn-Red Rank最高的前25%
    num_syn_core = int(len(df_sorted) * 0.25)
    syn_core_heads = all_head_uids[:num_syn_core]

    # 冗余核心：Syn-Red Rank最低的后25%
    red_core_heads = all_head_uids[-num_syn_core:]

    print(f"✅ 总头数: {len(all_head_uids)}")
    print(f"   协同核心: {len(syn_core_heads)} 个头 (Syn-Red Rank最高的前25%)")
    print(f"   冗余核心: {len(red_core_heads)} 个头 (Syn-Red Rank最低的后25%)")

    # 运行GSM8K准确率实验
    results = run_gsm8k_accuracy_experiment(
        model, tokenizer,
        syn_core_heads, red_core_heads, all_head_uids,
        noise_fraction=0.1,  # 自适应噪声：权重标准差的10%
        num_samples=100  # 使用100个样本快速测试
    )

    # 绘制Figure 4b
    print("\n" + "=" * 60)
    print("生成Figure 4b可视化")
    print("=" * 60)
    plot_figure4b(results)

    print("\n" + "=" * 60)
    print("✅ 分析完成！")
    print("=" * 60)

    # 打印关键发现
    print("\n" + "=" * 60)
    print("📊 关键发现 (25% 扰动)")
    print("=" * 60)

    baseline_acc = results['baseline_acc']
    syn_acc = results['syn_core_acc']
    red_acc = results['red_core_acc']
    random_acc = results['random_acc_mean']

    syn_drop = baseline_acc - syn_acc
    red_drop = baseline_acc - red_acc
    random_drop = baseline_acc - random_acc

    print(f"✅ GSM8K准确率对比:")
    print(f"   Baseline: {baseline_acc:.4f}")
    print(f"   协同核心: {syn_acc:.4f} (下降 {syn_drop:.4f}, {syn_drop/baseline_acc*100:.1f}%)")
    print(f"   冗余核心: {red_acc:.4f} (下降 {red_drop:.4f}, {red_drop/baseline_acc*100:.1f}%)")
    print(f"   随机: {random_acc:.4f} (下降 {random_drop:.4f}, {random_drop/baseline_acc*100:.1f}%)")

    print(f"\n🔍 验证论文发现:")
    # 协同核心应该准确率下降最大
    if syn_drop > red_drop:
        diff = syn_drop - red_drop
        print(f"   ✓ 协同核心扰动导致准确率下降更大")
        print(f"     协同核心 vs 冗余核心: {diff:.4f} ({diff/baseline_acc*100:.1f}%)")
    else:
        diff = red_drop - syn_drop
        print(f"   ✗ 未验证：冗余核心影响大于协同核心")
        print(f"     差异: {diff:.4f}")

    # 随机应该介于两者之间
    if max(syn_drop, red_drop) >= random_drop >= min(syn_drop, red_drop):
        print(f"   ✓ 随机扰动介于协同和冗余之间")
    else:
        print(f"   ⚠ 随机扰动不在预期范围")


if __name__ == "__main__":
    main()
