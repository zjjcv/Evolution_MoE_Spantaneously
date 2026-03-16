# Synergistic Core in LLMs

Replication of "A Brain-like Synergistic Core in LLMs Drives Behaviour and Learning" (Urbina-Rodriguez et al., 2026) on Gemma3-4B-Instruct model.

## Project Structure

```
├── src/                      # Main scripts
│   ├── figure4a_perturbation.py   # Behavior divergence with head perturbation
│   ├── figure4b_math_accuracy.py  # GSM8K accuracy with perturbed heads
│   ├── figure5_finetuning.py      # Fine-tuning synergistic vs redundant core
│   ├── activation_collection.py   # Collect activations for ΦID computation
│   └── test_inference.py          # Test model inference
│
├── utils/                    # Utilities
│   ├── compute_syn_red_rank.py    # Compute synergy-redundancy ranks
│   └── plot/                       # Visualization scripts
│       ├── figure3b_network.py     # Network graphs
│       ├── figure3c_metrics.py     # Network metrics
│       └── synergy_core.py
│
├── data/                     # Dataset files (not tracked)
├── results/                  # Experiment results (not tracked)
└── Gemma-3-4B-Instruct/      # Model weights (not tracked)
```

## Experiments

- **Figure 3b**: Network graphs showing synergistic vs redundant core connectivity
- **Figure 3c**: Network metrics (clustering, modularity, efficiency)
- **Figure 4a**: Behavior divergence measured by KL divergence
- **Figure 4b**: GSM8K accuracy with head perturbation
- **Figure 5**: Fine-tuning synergistic vs redundant core (SFT vs GRPO)

## Requirements

```bash
pip install torch transformers datasets numpy pandas matplotlib seaborn tqdm
pip install verl  # For GRPO fine-tuning
```

## Usage

### Compute Syn-Red Ranks

```bash
python utils/compute_syn_red_rank.py
```

### Run Figure Experiments

```bash
# Figure 3b: Network graphs
python utils/plot/figure3b_network.py

# Figure 3c: Network metrics
python utils/plot/figure3c_metrics.py

# Figure 4a: Perturbation analysis
python src/figure4a_perturbation.py

# Figure 4b: GSM8K accuracy
python src/figure4b_math_accuracy.py

# Figure 5: Fine-tuning
python src/figure5_finetuning.py
```

## Model

All experiments use **Gemma3-4B-Instruct** model.

Download from HuggingFace:
```bash
huggingface-cli download google/gemma-3-4b-it
```

## Dataset

Uses **GSM8K** dataset for math reasoning evaluation.

## Citation

```bibtex
@article{urbina2026synergistic,
  title={A Brain-like Synergistic Core in LLMs Drives Behaviour and Learning},
  author={Urbina-Rodriguez, Adrian and others},
  journal={arXiv preprint},
  year={2026}
}
```
