import os
from vllm import LLM, SamplingParams

def run_qwen_inference():
    # 1. 配置路径与模型参数
    model_path = "/data/zjj/Synergistic_Core/Qwen-3-8B-base"
    
    # 检查路径是否存在
    if not os.path.exists(model_path):
        print(f"❌ 错误: 未能在路径 {model_path} 找到权重文件")
        return

    # 2. 硬件加速配置
    # 4090 建议开启 flash_attn。Qwen-3-8B 虽然是 8B 模型，单卡 24G 显存够用，
    # 但你有 8 张卡，如果想测试多卡并行，可以设置 tensor_parallel_size。
    tp_size = 1  # 8B 模型单卡 4090 即可跑通，若需测试多卡联动可改为 2, 4 或 8
    
    print(f"🚀 正在加载模型 (TP Size: {tp_size})...")
    
    # 初始化 vLLM 引擎
    # gpu_memory_utilization: 控制显存占用比例，默认 0.9，4090 建议保留一点余地
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.85, 
        dtype="bfloat16",
        max_model_len=2048  # Qwen-3 建议使用 bf16
    )

    # 3. 设置采样参数 (Base 模型建议降低温度或设置 stop 词以免无限生成)
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=256,
        stop=["<|endoftext|>", "<|im_end|>"]
    )

    # 4. 准备测试 Prompt
    # 注意：Base 模型不是对话模型，它更倾向于续写
    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial Intelligence is changing the world by",
        "以‘今天天气不错’为开头写一段话："
    ]

    # 5. 执行推理
    print("\n" + "="*20 + " 推理结果 " + "="*20)
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\n📝 Prompt: {prompt}")
        print(f"🤖 Output: {generated_text}")
    print("\n" + "="*49)

if __name__ == "__main__":
    # 针对 4090 的潜在 P2P 问题，有时需要禁用 NCCL 的 P2P 传输（如果遇到卡死）
    # os.environ["NCCL_P2P_DISABLE"] = "1"
    # os.environ["NCCL_IB_DISABLE"] = "1"
    
    run_qwen_inference()