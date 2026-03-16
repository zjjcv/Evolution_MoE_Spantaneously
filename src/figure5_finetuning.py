import os
import re
import json
import math
import shlex
import random
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import torch
import torch.distributed as dist
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

from transformers import Trainer

class NoCheckpointTrainer(Trainer):
    """完全禁止训练过程中的 checkpoint 保存，避免 FSDP 聚合 optimizer state。"""

    def _save_checkpoint(self, model, trial):
        # 完全跳过 checkpoint 保存
        return

    def _save_optimizer_and_scheduler(self, output_dir):
        # 双保险：即使别的路径触发，也不保存 optimizer/scheduler
        return

    def _save_rng_state(self, output_dir):
        # 进一步避免 RNG state 保存
        return

# =========================
# Global config
# =========================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_PATH = "/data/zjj/Synergistic_Core/Gemma-3-4B-Instruct"
GSM8K_DATA_DIR = "/data/zjj/Synergistic_Core/data/gsm8k/main"
HEAD_SYN_RED_PATH = "./results/Gemma3-4B-Instruct/head_syn_red_ranks.csv"
OUTPUT_DIR = "./results/Gemma3-4B-Instruct/figure5_fixed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ExperimentConfig:
    approach: str              # 'synergistic', 'redundant', 'random'
    method: str                # 'sft', 'grpo'
    head_fraction: float = 0.5
    num_epochs: int = 3
    learning_rate: float = 5e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_samples: int = 500
    eval_samples: int = 100

    # GRPO-specific
    rollout_n: int = 4
    max_prompt_length: int = 512
    max_response_length: int = 256
    ppo_mini_batch_size: int = 16
    ppo_micro_batch_size_per_gpu: int = 1
    log_prob_micro_batch_size_per_gpu: int = 1
    gpu_memory_utilization: float = 0.5


# =========================
# Utilities
# =========================
def load_syn_red_rank_data(csv_path: str) -> pd.DataFrame:
    print(f"📊 Loading Syn-Red Rank data from {csv_path}")
    df = pd.read_csv(csv_path)
    df["UID"] = df["Layer"].astype(str) + "_" + df["Head"].astype(str)
    print(f"   Loaded {len(df)} heads")
    return df


def select_heads_for_ft(df: pd.DataFrame, config: ExperimentConfig) -> List[str]:
    df_sorted = df.sort_values(by="Syn_Red_Rank", ascending=False)
    total_heads = len(df_sorted)
    num_ft = int(total_heads * config.head_fraction)

    if config.approach == "synergistic":
        selected_uids = df_sorted.head(num_ft)["UID"].tolist()
        print(f"✅ Synergistic Core: selected top {num_ft} heads")
        print(f"   Mean Syn-Red Rank: {df_sorted.head(num_ft)['Syn_Red_Rank'].mean():.4f}")
    elif config.approach == "redundant":
        selected_uids = df_sorted.tail(num_ft)["UID"].tolist()
        print(f"✅ Redundant Core: selected bottom {num_ft} heads")
        print(f"   Mean Syn-Red Rank: {df_sorted.tail(num_ft)['Syn_Red_Rank'].mean():.4f}")
    elif config.approach == "random":
        all_uids = df_sorted["UID"].tolist()
        selected_uids = random.sample(all_uids, num_ft)
        selected_df = df[df["UID"].isin(selected_uids)]
        print(f"✅ Random Subset: selected {num_ft} heads")
        print(f"   Mean Syn-Red Rank: {selected_df['Syn_Red_Rank'].mean():.4f}")
    else:
        raise ValueError(f"Unknown approach: {config.approach}")

    selected_layers = sorted(set(int(uid.split("_")[0]) for uid in selected_uids))
    print(f"   Covered layers: {selected_layers[:10]}{' ...' if len(selected_layers) > 10 else ''}")
    print(f"   Num covered layers: {len(selected_layers)}")

    return selected_uids


def get_selected_layers_from_uids(selected_uids: List[str]) -> List[int]:
    return sorted(set(int(uid.split("_")[0]) for uid in selected_uids))


def freeze_model_for_layerwise_attention_ft(model, selected_layers: List[int]) -> None:
    """
    注意：这不是“真·head-level训练”。
    这里只解冻 selected_layers 中各层 self_attn 的 q/k/o 投影参数，
    其余全部冻结。
    """
    rank0_print(f"🔒 Freezing model; only q/k/o in selected layers are trainable")
    rank0_print(f"   Selected layers: {selected_layers}")

    frozen_params = 0
    ft_params = 0

    for name, param in model.named_parameters():
        requires_grad = False

        if "self_attn" in name and any(x in name for x in ["q_proj", "k_proj", "o_proj"]):
            parts = name.split(".")
            if "layers" in parts:
                layer_pos = parts.index("layers")
                layer_idx = int(parts[layer_pos + 1])
                if layer_idx in selected_layers:
                    requires_grad = True

        param.requires_grad = requires_grad

        if requires_grad:
            ft_params += param.numel()
        else:
            frozen_params += param.numel()

    total = frozen_params + ft_params
    ratio = 100.0 * ft_params / total if total > 0 else 0.0
    rank0_print(f"   Frozen params: {frozen_params:,}")
    rank0_print(f"   Trainable params: {ft_params:,} ({ratio:.4f}%)")

def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))

def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))

def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))

def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def is_main_process() -> bool:
    return get_rank() == 0

def dist_barrier():
    if is_dist():
        dist.barrier()

def rank0_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)

def load_gsm8k_train_test(
    train_parquet: str,
    test_parquet: str,
    max_train: Optional[int] = None,
    max_test: Optional[int] = None,
) -> Tuple[Dataset, Dataset]:
    print("📚 Loading GSM8K parquet files")

    train_df = pd.read_parquet(train_parquet)
    if max_train is not None:
        train_df = train_df.sample(n=min(max_train, len(train_df)), random_state=SEED)
    print(f"   Train samples: {len(train_df)}")

    test_df = pd.read_parquet(test_parquet)
    if max_test is not None:
        test_df = test_df.sample(n=min(max_test, len(test_df)), random_state=SEED)
    print(f"   Test samples: {len(test_df)}")

    return Dataset.from_pandas(train_df), Dataset.from_pandas(test_df)


def format_gsm8k_for_training(examples):
    texts = []
    for question, answer in zip(examples["question"], examples["answer"]):
        texts.append(f"Question: {question}\nAnswer: {answer}")
    return {"text": texts}


def extract_final_answer(text: str) -> Optional[str]:
    if "####" in text:
        return text.split("####")[-1].strip()
    return None


def check_gsm8k_answer_correctness(generated: str, reference: str) -> bool:
    gen_answer = extract_final_answer(generated)
    ref_answer = extract_final_answer(reference)
    if gen_answer is None or ref_answer is None:
        return False

    gen_clean = gen_answer.replace(",", "").replace(" ", "").lower()
    ref_clean = ref_answer.replace(",", "").replace(" ", "").lower()
    return gen_clean == ref_clean


def evaluate_gsm8k_accuracy(model, tokenizer, test_dataset: Dataset) -> float:
    if not is_main_process():
        return 0.0

    rank0_print(f"📊 Evaluating on {len(test_dataset)} samples")
    model.eval()

    correct = 0
    total = 0

    device = next(model.parameters()).device

    for sample in tqdm(test_dataset, desc="Evaluating"):
        prompt = f"Question: {sample['question']}\nAnswer:"
        reference_answer = sample["answer"]

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if check_gsm8k_answer_correctness(generated, reference_answer):
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0.0
    rank0_print(f"   Accuracy: {acc:.4f} ({correct}/{total})")
    return acc


def load_model_and_tokenizer(model_path: str):
    local_rank = get_local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    rank0_print(f"📦 Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    return model, tokenizer


def save_trainable_model_snapshot(model, tokenizer, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    print(f"💾 Saving model snapshot to {save_dir}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


def write_custom_reward_function(reward_py_path: str):
    reward_code = r'''
import re

def _extract_num(s: str):
    if s is None:
        return None
    if "####" in s:
        s = s.split("####")[-1]
    s = s.strip()
    m = re.findall(r"-?\d[\d,]*\.?\d*", s)
    if not m:
        return None
    return m[-1].replace(",", "")

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    verl custom reward API:
      compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float
    """
    if solution_str is None:
        return 0.0

    pred = _extract_num(solution_str)
    gt = _extract_num(ground_truth)

    # 1) exact answer match
    if pred is not None and gt is not None and pred == gt:
        return 1.0

    # 2) has #### but wrong
    if "####" in solution_str:
        return 0.1

    # 3) otherwise zero
    return 0.0
'''
    with open(reward_py_path, "w", encoding="utf-8") as f:
        f.write(reward_code.strip() + "\n")
    print(f"📝 Wrote custom reward fn to {reward_py_path}")


def prepare_grpo_parquet(train_dataset: Dataset, test_dataset: Dataset, train_path: str, val_path: str):
    train_records = []
    for sample in train_dataset:
        train_records.append({
            "prompt": f"Question: {sample['question']}\nAnswer:",
            "ground_truth": sample["answer"],
            "data_source": "gsm8k",
        })

    val_records = []
    for sample in test_dataset:
        val_records.append({
            "prompt": f"Question: {sample['question']}\nAnswer:",
            "ground_truth": sample["answer"],
            "data_source": "gsm8k",
        })

    pd.DataFrame(train_records).to_parquet(train_path, index=False)
    pd.DataFrame(val_records).to_parquet(val_path, index=False)

    print(f"💾 Saved GRPO train parquet: {train_path}")
    print(f"💾 Saved GRPO val parquet:   {val_path}")


def find_latest_global_step(ckpt_root: str) -> Optional[str]:
    ckpt_root = Path(ckpt_root)
    if not ckpt_root.exists():
        return None

    pattern = re.compile(r"global_step_(\d+)")
    candidates = []
    for p in ckpt_root.iterdir():
        if p.is_dir():
            m = pattern.fullmatch(p.name)
            if m:
                candidates.append((int(m.group(1)), str(p)))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def merge_fsdp_checkpoint(local_dir: str, target_dir: str):
    cmd = [
        "python", "-m", "verl.model_merger", "merge",
        "--backend", "fsdp",
        "--local_dir", local_dir,
        "--target_dir", target_dir,
    ]
    print("🔧 Merging FSDP checkpoint:")
    print("   " + " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)


# =========================
# SFT
# =========================
def run_sft_finetuning(
    model,
    tokenizer,
    train_dataset: Dataset,
    test_dataset: Dataset,
    config: ExperimentConfig
) -> float:
    rank0_print(f"\n{'=' * 60}")
    rank0_print(f"🎯 Running SFT - {config.approach.upper()} (FSDP)")
    rank0_print(f"{'=' * 60}")

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    train_dataset = train_dataset.map(
        format_gsm8k_for_training,
        batched=True,
        remove_columns=train_dataset.column_names
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=384,
            padding="max_length"
        )

    tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])

    fsdp_config = {
        "transformer_layer_cls_to_wrap": ["Gemma3DecoderLayer"],
        "use_orig_params": True,
        "sync_module_states": True,
        "cpu_ram_efficient_loading": False,
        "activation_checkpointing": True,
        "limit_all_gathers": True,
        "forward_prefetch": False,
        "backward_prefetch": "backward_pre",
    }

    common_kwargs = dict(
        output_dir=f"{OUTPUT_DIR}/sft_{config.approach}",
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=0.01,
        warmup_steps=20,
        logging_steps=10,
        save_strategy="no",
        save_only_model=True,
        fp16=False,
        bf16=torch.cuda.is_bf16_supported(),
        seed=SEED,
        report_to="none",
        optim="adafactor",
        ddp_timeout=1800,
        fsdp="full_shard auto_wrap",
        fsdp_config=fsdp_config,
    )

    try:
        training_args = TrainingArguments(
            **common_kwargs,
            eval_strategy="no",
        )
    except TypeError:
        training_args = TrainingArguments(
            **common_kwargs,
            evaluation_strategy="no",
        )

    trainer = NoCheckpointTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    rank0_print("🚀 Starting SFT with FSDP...")
    trainer.train()

    dist_barrier()

    if is_main_process():
        rank0_print("💾 Saving final model only...")
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

    dist_barrier()

    if not is_main_process():
        return 0.0

    rank0_print("📦 Reloading saved checkpoint for rank0 evaluation...")
    eval_model = AutoModelForCausalLM.from_pretrained(
        training_args.output_dir,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if torch.cuda.is_available():
        eval_model = eval_model.to("cuda:0")

    rank0_print("📊 Evaluating after SFT...")
    accuracy = evaluate_gsm8k_accuracy(eval_model, tokenizer, test_dataset)

    del eval_model
    torch.cuda.empty_cache()
    return accuracy


# =========================
# GRPO via verl.main_ppo
# =========================
def run_grpo_finetuning_verl(
    selected_model_dir: str,
    tokenizer,
    train_dataset: Dataset,
    test_dataset: Dataset,
    config: ExperimentConfig,
) -> float:
    rank0_print(f"\n{'=' * 60}")
    rank0_print(f"🎯 Running GRPO - {config.approach.upper()}")
    rank0_print(f"{'=' * 60}")

    exp_dir = os.path.join(OUTPUT_DIR, f"grpo_{config.approach}")
    os.makedirs(exp_dir, exist_ok=True)

    train_parquet = os.path.join(exp_dir, "train.parquet")
    val_parquet = os.path.join(exp_dir, "val.parquet")
    reward_py = os.path.join(exp_dir, "gsm8k_reward.py")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")

    prepare_grpo_parquet(train_dataset, test_dataset, train_parquet, val_parquet)
    write_custom_reward_function(reward_py)

    # 关键：这里使用保存下来的“冻结后快照”作为 verl 的 model.path
    cmd = [
        "python", "-m", "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        f"data.train_files={train_parquet}",
        f"data.val_files={val_parquet}",
        "data.prompt_key=prompt",
        "data.max_prompt_length=512",
        f"data.max_response_length={config.max_response_length}",
        f"data.train_batch_size={config.batch_size * config.gradient_accumulation_steps * config.rollout_n}",
        f"data.val_batch_size={max(1, len(test_dataset))}",
        "data.filter_overlong_prompts=True",
        "data.truncation=right",

        f"actor_rollout_ref.model.path={selected_model_dir}",
        "actor_rollout_ref.model.use_remove_padding=True",
        f"actor_rollout_ref.actor.optim.lr={config.learning_rate}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={config.ppo_mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={config.ppo_micro_batch_size_per_gpu}",

        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.n={config.rollout_n}",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={config.gpu_memory_utilization}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={config.log_prob_micro_batch_size_per_gpu}",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={config.log_prob_micro_batch_size_per_gpu}",

        "algorithm.use_kl_in_reward=False",
        "trainer.critic_warmup=0",
        "trainer.val_before_train=False",
        "trainer.test_freq=1",
        "trainer.save_freq=1",
        "trainer.resume_mode=disable",
        f"trainer.total_epochs={config.num_epochs}",
        "trainer.logger=[console]",
        "trainer.nnodes=1",
        "trainer.n_gpus_per_node=1",
        f"trainer.default_local_dir={ckpt_dir}",
        "trainer.default_hdfs_dir=null",

        f"custom_reward_function.path={reward_py}",
        "custom_reward_function.name=compute_score",
    ]

    rank0_print("🚀 Launching verl GRPO command:")
    rank0_print("   " + " ".join(shlex.quote(x) for x in cmd))

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"verl main_ppo failed with return code {result.returncode}")

    latest_ckpt = find_latest_global_step(ckpt_dir)
    if latest_ckpt is None:
        raise RuntimeError(f"No global_step_* checkpoint found under {ckpt_dir}")

    print(f"📁 Latest checkpoint: {latest_ckpt}")

    merged_dir = os.path.join(latest_ckpt, "huggingface")
    if not os.path.exists(os.path.join(merged_dir, "config.json")):
        merge_fsdp_checkpoint(latest_ckpt, merged_dir)

    print(f"📦 Loading merged HF checkpoint from {merged_dir}")
    tuned_model = AutoModelForCausalLM.from_pretrained(
        merged_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    print("📊 Evaluating GRPO model...")
    acc = evaluate_gsm8k_accuracy(tuned_model, tokenizer, test_dataset)

    del tuned_model
    torch.cuda.empty_cache()

    return acc


# =========================
# Main experiment
# =========================
def run_experiment(
    config: ExperimentConfig,
    model_path: str,
    head_syn_red_path: str,
    train_parquet: str,
    test_parquet: str,
) -> Dict:
    print(f"\n{'=' * 70}")
    print(f"🧪 EXPERIMENT: {config.approach.upper()} - {config.method.upper()}")
    print(f"{'=' * 70}")

    model, tokenizer = load_model_and_tokenizer(model_path)

    # 读取头排序，转成 layer-level 覆盖集
    df_syn_red = load_syn_red_rank_data(head_syn_red_path)
    selected_uids = select_heads_for_ft(df_syn_red, config)
    selected_layers = get_selected_layers_from_uids(selected_uids)
    freeze_model_for_layerwise_attention_ft(model, selected_layers)

    # 保存一份快照给后续 SFT / GRPO 使用
    selected_model_dir = os.path.join(OUTPUT_DIR, f"model_snapshot_{config.approach}_{config.method}")
    save_trainable_model_snapshot(model, tokenizer, selected_model_dir)

    train_dataset, test_dataset = load_gsm8k_train_test(
        train_parquet=train_parquet,
        test_parquet=test_parquet,
        max_train=config.max_samples,
        max_test=config.eval_samples,
    )

    if config.method == "sft":
        accuracy = run_sft_finetuning(model, tokenizer, train_dataset, test_dataset, config)

    elif config.method == "grpo":
        if is_main_process():
            save_trainable_model_snapshot(model, tokenizer, selected_model_dir)
        dist_barrier()

        del model
        torch.cuda.empty_cache()

        accuracy = run_grpo_finetuning_verl(
            selected_model_dir=selected_model_dir,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            config=config,
        )
    else:
        raise ValueError(f"Unknown method: {config.method}")

    torch.cuda.empty_cache()

    return {
        "approach": config.approach,
        "method": config.method,
        "accuracy": accuracy,
    }


def plot_figure5_results(results_df: pd.DataFrame):
    print("\n📊 Creating visualization...")

    sft_data = results_df[results_df["method"] == "sft"].pivot(index="approach", columns="method", values="accuracy")
    grpo_data = results_df[results_df["method"] == "grpo"].pivot(index="approach", columns="method", values="accuracy")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    approaches = ["synergistic", "redundant", "random"]
    labels = ["Synergistic\nCore", "Redundant\nCore", "Random\nSubset"]
    colors = ["#2ecc71", "#e74c3c", "#95a5a6"]

    sft_accs = [sft_data.loc[a, "sft"] if a in sft_data.index else 0 for a in approaches]
    bars1 = axes[0].bar(labels, sft_accs, color=colors, alpha=0.85, edgecolor="black", linewidth=1.2)
    axes[0].set_title("SFT", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("GSM8K Accuracy")
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(axis="y", alpha=0.3)
    for bar, acc in zip(bars1, sft_accs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{acc:.3f}",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")

    grpo_accs = [grpo_data.loc[a, "grpo"] if a in grpo_data.index else 0 for a in approaches]
    bars2 = axes[1].bar(labels, grpo_accs, color=colors, alpha=0.85, edgecolor="black", linewidth=1.2)
    axes[1].set_title("GRPO", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("GSM8K Accuracy")
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(axis="y", alpha=0.3)
    for bar, acc in zip(bars2, grpo_accs):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{acc:.3f}",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.suptitle("Figure 5: Layer-wise Attention FT for Selected Head Coverage", fontsize=15, fontweight="bold")
    plt.tight_layout()

    out_png = os.path.join(OUTPUT_DIR, "figure5_results.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✅ Figure saved to {out_png}")

    print("\n📊 Results Summary:")
    print("=" * 60)
    print(f"{'Approach':<15} {'SFT':<10} {'GRPO':<10} {'Delta':<10}")
    print("-" * 60)
    for approach in approaches:
        sft_val = sft_data.loc[approach, "sft"] if approach in sft_data.index else 0
        grpo_val = grpo_data.loc[approach, "grpo"] if approach in grpo_data.index else 0
        delta = grpo_val - sft_val
        print(f"{approach.capitalize():<15} {sft_val:<10.4f} {grpo_val:<10.4f} {delta:<10.4f}")
    print("=" * 60)


def main():
    rank0_print("=" * 70)
    rank0_print("Figure 5: Fine-tuning Synergistic vs Redundant Core (Fixed)")
    rank0_print("=" * 70)

    train_parquet = os.path.join(GSM8K_DATA_DIR, "train-00000-of-00001.parquet")
    test_parquet = os.path.join(GSM8K_DATA_DIR, "test-00000-of-00001.parquet")

    if not os.path.exists(train_parquet):
        raise FileNotFoundError(f"Train file not found: {train_parquet}")
    if not os.path.exists(test_parquet):
        raise FileNotFoundError(f"Test file not found: {test_parquet}")
    if not os.path.exists(HEAD_SYN_RED_PATH):
        raise FileNotFoundError(f"Head syn-red data not found: {HEAD_SYN_RED_PATH}")

    experiments = [
        ExperimentConfig(
            approach="synergistic",
            method="sft",
            num_epochs=1,
            max_samples=500,
            eval_samples=100,
            batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
        ),
    ]

    results = []
    for exp in experiments:
        try:
            result = run_experiment(
                config=exp,
                model_path=MODEL_PATH,
                head_syn_red_path=HEAD_SYN_RED_PATH,
                train_parquet=train_parquet,
                test_parquet=test_parquet,
            )
            results.append(result)
            print(f"\n✅ {exp.approach.upper()} - {exp.method.upper()}: {result['accuracy']:.4f}")
        except Exception as e:
            print(f"\n❌ Error in {exp.approach} - {exp.method}: {e}")
            import traceback
            traceback.print_exc()

    if is_main_process():
        results_df = pd.DataFrame(results)
        results_csv = os.path.join(OUTPUT_DIR, "figure5_results.csv")
        results_df.to_csv(results_csv, index=False)
        print(f"\n💾 Results saved to {results_csv}")

        if len(results_df) > 0:
            plot_figure5_results(results_df)

        print("\n" + "=" * 70)
        print("✅ All experiments completed")
        print("=" * 70)


if __name__ == "__main__":
    main()