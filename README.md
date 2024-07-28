# llm-train-scripts

Short simple language model training scripts using Hugging Face's `transformers` library. These scripts are intended to be used as a starting point for training language models on custom datasets.

## Scripts

- `train_clm_eli5.py`: Train a causal language model on the ELI5 dataset
- `train_clm_eli5_loop.py`: Train a causal language model on the ELI5 dataset with a explicit training loop
- `train_clm_eli5_char.py`: Train a causal language model on the ELI5 dataset with character-level tokenization
- `train_clm_shakespeare_char.py`: Train a causal language model on the Shakespeare dataset with character-level tokenization

## Quickstart

1. Clone repo, install dependencies, and run script

```bash
git clone
cd llm-train-scripts
pip install -r requirements.txt
python train_clm_shakespeare_char.py
```

Sample output:

```

```
