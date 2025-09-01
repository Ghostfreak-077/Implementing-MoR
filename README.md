# Implementing-MoR

This repository contains the code and resources for my own simpler implementation of the Mixture of Recursions (MoR) model for efficient computation. The project tries to benchmark the MoR model against the LLaMA-2 model on the Wikitext-2 dataset.

## Requirements
You can install the required packages using pip:
```
pip install -r requirements.txt
```
## Usage
Current training script is located in `scripts/train_script.py`. You can run the training script using:
```
python scripts/train_script.py
```

## References
Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation; _paper by_ Sangmin Bae, Yujin Kim, Reza Bayat, Sungnyun Kim, Jiyoun Ha, Tal Schuster, Adam Fisch, Hrayr Harutyunyan, Ziwei Ji, Aaron Courville and Se-Young Yun; year: 2025; [Arxiv pre-print](https://arxiv.org/abs/2507.10524); [github](https://github.com/raymin0223/mixture_of_recursions)