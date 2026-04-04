"""
Convert Meta Llama 3.1 original format to HuggingFace format.

Input:  original/consolidated.00.pth + original/params.json + original/tokenizer.model
Output: config.json + model.safetensors + tokenizer files (in parent directory)

Usage:
    python src/convert_llama_to_hf.py
"""

import os
import json
import torch
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer

MODEL_DIR = "/root/data1/zjj/Neurlps2026/Checkpoints/Meta-Llama-3.1-8B-Instruct"
ORIGINAL_DIR = os.path.join(MODEL_DIR, "original")


def convert():
    print("=" * 60)
    print("Converting Meta Llama 3.1 to HuggingFace format")
    print("=" * 60)

    # Load params
    with open(os.path.join(ORIGINAL_DIR, "params.json")) as f:
        params = json.load(f)

    print(f"\nModel params: {json.dumps(params, indent=2)}")

    # Create HF config
    # Llama 3.1 8B: intermediate_size = 14336
    config = LlamaConfig(
        vocab_size=params["vocab_size"],
        hidden_size=params["dim"],
        intermediate_size=14336,
        num_hidden_layers=params["n_layers"],
        num_attention_heads=params["n_heads"],
        num_key_value_heads=params["n_kv_heads"],
        max_position_embeddings=131072,
        rms_norm_eps=params["norm_eps"],
        rope_theta=params["rope_theta"],
        tie_word_embeddings=False,
    )

    print(f"\nHF Config created:")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_layers: {config.num_hidden_layers}")
    print(f"  num_heads: {config.num_attention_heads}")
    print(f"  num_kv_heads: {config.num_key_value_heads}")
    print(f"  intermediate_size: {config.intermediate_size}")

    # Load original weights
    print(f"\nLoading original weights...")
    state_dict = torch.load(
        os.path.join(ORIGINAL_DIR, "consolidated.00.pth"),
        map_location="cpu",
        weights_only=True
    )

    # Remap keys: Meta format -> HF format
    print("Remapping keys...")
    new_state_dict = {}
    key_mapping = {
        "tok_embeddings.weight": "model.embed_tokens.weight",
        "output.weight": "lm_head.weight",
        "norm.weight": "model.norm.weight",
    }

    for old_key, value in state_dict.items():
        if old_key in key_mapping:
            new_key = key_mapping[old_key]
        elif old_key.startswith("layers."):
            # e.g., layers.0.attention.wq.weight -> model.layers.0.self_attn.q_proj.weight
            parts = old_key.split(".")
            layer_idx = parts[1]
            component = ".".join(parts[2:])

            sub_mapping = {
                "attention.wq.weight": f"model.layers.{layer_idx}.self_attn.q_proj.weight",
                "attention.wk.weight": f"model.layers.{layer_idx}.self_attn.k_proj.weight",
                "attention.wv.weight": f"model.layers.{layer_idx}.self_attn.v_proj.weight",
                "attention.wo.weight": f"model.layers.{layer_idx}.self_attn.o_proj.weight",
                "feed_forward.w1.weight": f"model.layers.{layer_idx}.mlp.gate_proj.weight",
                "feed_forward.w2.weight": f"model.layers.{layer_idx}.mlp.down_proj.weight",
                "feed_forward.w3.weight": f"model.layers.{layer_idx}.mlp.up_proj.weight",
                "attention_norm.weight": f"model.layers.{layer_idx}.input_layernorm.weight",
                "ffn_norm.weight": f"model.layers.{layer_idx}.post_attention_layernorm.weight",
            }

            new_key = sub_mapping.get(component)
            if new_key is None:
                print(f"  WARNING: Unknown key mapping: {old_key}")
                continue
        else:
            print(f"  WARNING: Unknown top-level key: {old_key}")
            continue

        new_state_dict[new_key] = value

    print(f"  Mapped {len(new_state_dict)} keys")

    # Create model with config and load weights
    print("\nCreating HuggingFace model...")
    model = LlamaForCausalLM(config)
    model.load_state_dict(new_state_dict, strict=True)
    print("  Weights loaded successfully (strict=True)")

    # Save config
    config_path = os.path.join(MODEL_DIR, "config.json")
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"  Saved: {config_path}")

    # Save model as safetensors
    print("\nSaving model as safetensors...")
    model.save_pretrained(MODEL_DIR, safe_serialization=True)
    print(f"  Saved to: {MODEL_DIR}")

    # Save tokenizer (copy tokenizer.model to model dir, then use AutoTokenizer)
    print("\nSaving tokenizer...")
    import shutil
    # Copy tokenizer.model to model dir
    tokenizer_model_src = os.path.join(ORIGINAL_DIR, "tokenizer.model")
    tokenizer_model_dst = os.path.join(MODEL_DIR, "tokenizer.model")
    shutil.copy2(tokenizer_model_src, tokenizer_model_dst)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False, legacy=False)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"  Saved to: {MODEL_DIR}")

    # Verify
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    for fname in ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]:
        fpath = os.path.join(MODEL_DIR, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  {fname}: {size:.1f} MB")
        else:
            print(f"  {fname}: NOT FOUND")

    print("\nDone!")


if __name__ == "__main__":
    convert()
