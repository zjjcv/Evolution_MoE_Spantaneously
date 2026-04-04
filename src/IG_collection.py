"""
Collect layer sensitivity to input tokens using perturbation-based attribution.

This method measures how much each layer's output changes when each input token
is perturbed. This is simpler and more stable than gradient-based methods.

Method:
    For each token position t:
        1. Get baseline layer outputs
        2. Perturb token t's embedding
        3. Measure change in each layer's output

Output: CSV file with columns: layer, token_position, token, ig_value

Usage:
    python src/IG_collection.py
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Configuration
MODEL_PATH = os.environ.get("MODEL_PATH", "/root/data1/zjj/Neurlps2026/Checkpoints/Meta-Llama-3.1-8B-Instruct")
GSM8K_DATA_DIR = "/root/data1/zjj/Neurlps2026/Dataset/gsm8k"
OUTPUT_DIR = "/root/data1/zjj/Neurlps2026/Results/Llama-3.1-8B-Instruct/data/IG_Relative/IG"

OUTPUT_CSV = os.path.join(OUTPUT_DIR, "IG.csv")


class LayerSensitivityComputer:
    """Compute layer sensitivity to input tokens using perturbation."""

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.num_layers = model.config.num_hidden_layers
        self.hidden_size = model.config.hidden_size

    def collect_all_sensitivity(self, question_text: str, max_tokens: int = 50) -> pd.DataFrame:
        """Collect sensitivity values for all layers and all tokens.

        Args:
            question_text: Input question text
            max_tokens: Maximum number of tokens to analyze

        Returns:
            DataFrame with columns: layer, token_position, token, sensitivity
        """
        print(f"\nTokenizing and processing question...")

        # Tokenize
        inputs = self.tokenizer(question_text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        tokens = [self.tokenizer.decode([tid]) for tid in input_ids[0].cpu().numpy()]

        seq_len = min(input_ids.shape[1], max_tokens)
        input_ids = input_ids[:, :seq_len]
        tokens = tokens[:seq_len]

        print(f"  Sequence length: {seq_len}")
        print(f"  Tokens: {' '.join(tokens[:10])}...")

        results = []

        # Get baseline outputs for each layer
        print("  Computing baseline layer outputs...")
        baseline_outputs = self._get_layer_outputs(input_ids)

        # For each token position, perturb and measure change
        print("  Computing sensitivity for each token...")
        for token_pos in range(seq_len):
            if token_pos % 10 == 0:
                print(f"    Token {token_pos}/{seq_len}...")

            # Perturb this token's embedding by adding noise
            perturbed_outputs = self._get_perturbed_outputs(input_ids, token_pos)

            # Compute sensitivity for each layer
            for layer_idx in range(self.num_layers):
                baseline = baseline_outputs[layer_idx]  # [seq_len, hidden]
                perturbed = perturbed_outputs[layer_idx]  # [seq_len, hidden]

                # Measure change using L2 distance of the entire layer output
                # This captures how much the layer's representation changes
                # when this token is perturbed
                baseline_flat = baseline.flatten()
                perturbed_flat = perturbed.flatten()

                sensitivity = np.linalg.norm(perturbed_flat - baseline_flat)

                results.append({
                    'layer': layer_idx,
                    'token_position': token_pos,
                    'token': tokens[token_pos],
                    'ig_value': float(sensitivity)
                })

        return pd.DataFrame(results)

    def _get_text_model(self):
        """Get the text model (model for Llama, language_model for Gemma3, etc.)."""
        # Llama: model.model is the core transformer
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model
        if hasattr(self.model, 'language_model'):
            return self.model.language_model
        return self.model

    def _get_embed_tokens(self):
        """Get the embedding layer."""
        # Llama: model.model.embed_tokens
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            return self.model.model.embed_tokens
        # Gemma3: model.model.language_model.embed_tokens
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'language_model') and hasattr(self.model.model.language_model, 'embed_tokens'):
            return self.model.model.language_model.embed_tokens
        return self.model.get_input_embeddings()

    def _get_layer_outputs(self, input_ids: torch.Tensor) -> dict:
        """Get layer outputs for the given input.

        Args:
            input_ids: Input token IDs [batch, seq_len]

        Returns:
            Dict mapping layer_idx -> output array [seq_len, hidden]
        """
        outputs = {}

        # Run model and get hidden states
        with torch.no_grad():
            model_output = self.model(input_ids, output_hidden_states=True,
                                      return_dict=True)
            hidden_states = model_output.hidden_states  # tuple of [batch, seq_len, hidden]

        # Convert to numpy (skip embedding layer, start from layer 0)
        # For Gemma3, hidden_states[0] is embedding, hidden_states[1] is layer 0 output, etc.
        for layer_idx in range(self.num_layers):
            hidden = hidden_states[layer_idx + 1]
            outputs[layer_idx] = hidden[0, :, :].cpu().float().numpy()

        return outputs

    def _get_perturbed_outputs(self, input_ids: torch.Tensor,
                               perturb_pos: int) -> dict:
        """Get layer outputs after perturbing a token's embedding.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            perturb_pos: Position to perturb

        Returns:
            Dict mapping layer_idx -> output array [seq_len, hidden]
        """
        # Get embeddings
        embed_tokens = self._get_embed_tokens()
        embeds = embed_tokens(input_ids)  # [batch, seq_len, hidden]

        # Add noise to the perturbed position
        noise = torch.randn_like(embeds[:, perturb_pos, :]) * 0.1  # Perturbation magnitude
        embeds[:, perturb_pos, :] = embeds[:, perturb_pos, :] + noise

        # Run model with perturbed embeddings
        outputs = {}

        with torch.no_grad():
            model_output = self.model(inputs_embeds=embeds, output_hidden_states=True,
                                      return_dict=True)
            hidden_states = model_output.hidden_states

        # Convert to numpy (skip embedding layer)
        for layer_idx in range(self.num_layers):
            hidden = hidden_states[layer_idx + 1]
            outputs[layer_idx] = hidden[0, :, :].cpu().float().numpy()

        return outputs


def load_first_question(data_dir: str) -> str:
    """Load the first question from GSM8K test set.

    Args:
        data_dir: GSM8K data directory

    Returns:
        First question text
    """
    test_file = os.path.join(data_dir, "json", "test.json")

    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    first_question = data[0]

    if isinstance(first_question, dict):
        question_text = first_question.get('question', '')
    else:
        question_text = str(first_question)

    return question_text


def create_prompt(question: str) -> str:
    """Create prompt for GSM8K question.

    Args:
        question: Question text

    Returns:
        Formatted prompt
    """
    if "Question:" in question:
        return question
    else:
        return f"Question: {question}\nAnswer:"


def main():
    """Main execution function."""
    print("=" * 60)
    print("Layer Sensitivity Collection (Perturbation-Based)")
    print("=" * 60)

    # Load question
    print("\nLoading first GSM8K question...")
    question = load_first_question(GSM8K_DATA_DIR)
    prompt = create_prompt(question)
    print(f"  Prompt: {prompt[:100]}...")

    # Load model
    print(f"\nLoading model from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create sensitivity computer
    computer = LayerSensitivityComputer(model, tokenizer, device)

    # Collect sensitivity data
    print("\n" + "=" * 60)
    print("Collecting layer sensitivity...")
    print("=" * 60)

    ig_df = computer.collect_all_sensitivity(prompt, max_tokens=50)

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ig_df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"\nTotal records: {len(ig_df):,}")
    print(f"Layers: {ig_df['layer'].nunique()}")
    print(f"Token positions: {ig_df['token_position'].nunique()}")
    print(f"\nSensitivity range: [{ig_df['ig_value'].min():.6f}, {ig_df['ig_value'].max():.6f}]")
    print(f"Mean sensitivity: {ig_df['ig_value'].mean():.6f}")

    print("\n" + "=" * 60)
    print(f"Saved to: {OUTPUT_CSV}")
    print("All Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
