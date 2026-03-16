import os
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def main():
    # 1. 配置路径与参数
    model_path = "/data/zjj/Synergistic_Core/Gemma-3-4B-Instruct"
    save_dir = "./results/Gemma3-4B-Instruct"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "atten_activation_L2.csv")

    print("🚀 正在加载 Gemma-3-4B-Instruct 模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.eval()

    # 获取模型配置（Gemma3使用text_config）
    config = model.config
    if hasattr(config, 'text_config'):
        text_config = config.text_config
        num_layers = text_config.num_hidden_layers
        hidden_size = text_config.hidden_size
    else:
        num_layers = len(model.language_model.layers)
        hidden_size = config.hidden_size if hasattr(config, 'hidden_size') else 2560

    # 从第一层获取注意力头配置（GQA结构）
    first_layer_attn = model.language_model.layers[0].self_attn
    head_dim = first_layer_attn.head_dim
    q_proj_out = first_layer_attn.q_proj.out_features
    num_heads = q_proj_out // head_dim  # query头数

    print(f"✅ 模型配置: {num_layers} 层, {num_heads} query头/层, head_dim={head_dim}")

    # 2. 录入论文中的 6 个类别 Prompts，每个类别取第一条 (Table 2 & Table 3)
    prompt_dict = {
        "Syntax_and_Grammar": [
            "Correct the error: He go to school every day."
        ],
        "Part_of_Speech": [
            "Identify the parts of speech in the sentence: Quickly, the agile cat climbed the tall tree."
        ],
        "Numerical_Reasoning": [
            "If you have 15 apples and you give away 5, how many do you have left?"
        ],
        "Common_Sense": [
            "If it starts raining while the sun is shining, what weather phenomenon might you expect to see?"
        ],
        "Abstract_Reasoning": [
            "Imagine a future where humans have evolved to live underwater. Describe the adaptations they might develop."
        ],
        "Emotional_Intelligence": [
            "Write a dialogue between two characters where one comforts the other after a loss, demonstrating empathy."
        ]
    }

    # 将字典展平为列表
    all_prompts = []
    for category, p_list in prompt_dict.items():
        for p in p_list:
            all_prompts.append({"category": category, "text": p})

    # 激活缓冲区：(layer, head) -> list of L2 norms
    activation_buffer = {(i, j): [] for i in range(num_layers) for j in range(num_heads)}

    # 3. 定义并注册 PyTorch Forward Hook
    # Hook在q_proj输出，计算每个注意力头的L2范数
    def get_activation_hook(layer_idx):
        def hook(module, input, output):
            # output是q_proj的输出: [batch, seq, num_heads * head_dim]
            # reshape为 [batch, seq, num_heads, head_dim]
            batch, seq_len, _ = output.shape
            reshaped = output.view(batch, seq_len, num_heads, head_dim)

            # 取最后一个token的所有头
            last_token_heads = reshaped[:, -1, :, :]  # [batch, num_heads, head_dim]

            # 计算每个头的L2范数
            head_l2_norms = torch.linalg.norm(last_token_heads, dim=-1)  # [batch, num_heads]

            # 存储每个头的L2范数
            for head_idx in range(num_heads):
                activation_buffer[(layer_idx, head_idx)].append(
                    head_l2_norms[0, head_idx].cpu().float().item()
                )
        return hook

    hooks = []
    for i in range(num_layers):
        # Hook在self_attn.q_proj的输出上，获取每个注意力头的激活
        hook_handle = model.language_model.layers[i].self_attn.q_proj.register_forward_hook(get_activation_hook(i))
        hooks.append(hook_handle)

    # 4. 开始推理并收集数据
    print(f"🧠 开始处理全部 {len(all_prompts)} 个 Prompts...")
    all_records = []

    for prompt_id, prompt_info in enumerate(tqdm(all_prompts, desc="推理进度")):
        prompt_text = prompt_info["text"]
        category = prompt_info["category"]
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        # 清空激活缓冲区
        for layer in range(num_layers):
            for head in range(num_heads):
                activation_buffer[(layer, head)].clear()

        # 生成 100 个 token
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=100,
                min_new_tokens=100,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=None,
                use_cache=True
            )

        # 提取刚刚生成的数据（每个头一条记录）
        for layer in range(num_layers):
            for head in range(num_heads):
                # 获取最后 100 步的时间序列数据（L2范数标量）
                head_data = activation_buffer[(layer, head)][-100:]

                # 每个头一条记录
                record = {
                    "Prompt_ID": prompt_id,
                    "Category": category,
                    "Layer": layer,
                    "Head": head
                }
                for step_idx, step_val in enumerate(head_data):
                    record[f"Step_{step_idx+1}"] = step_val

                all_records.append(record)

    for h in hooks:
        h.remove()

    # 5. 保存为 CSV
    print(f"💾 正在保存数据至 {save_path} ...")
    df = pd.DataFrame(all_records)
    df.to_csv(save_path, index=False)
    print("✅ 提取完成！")

if __name__ == "__main__":
    main()