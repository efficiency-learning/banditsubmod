# Bandit-Guided Submodular Curriculum for Adaptive Subset Selection

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://neurips.cc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Official implementation of **"Bandit-Guided Submodular Curriculum for Adaptive Subset Selection"** accepted at **NeurIPS 2025**.

**Authors:** Prateek Chanda, Prayas Agrawal, Saral Sureka, Lokesh Reddy Polu, Atharv Kshirsagar, Ganesh Ramakrishnan

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Experiments](#experiments)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

> **New to the project?** Check out our [Quick Start Guide](QUICKSTART.md) for a 5-minute introduction!

## Overview

This repository contains the implementation of a novel bandit-guided submodular curriculum learning framework for adaptive data subset selection during training of deep learning models. Our method combines **multi-armed bandits** with **submodular optimization** to create an adaptive curriculum that selects informative data subsets throughout training, significantly reducing computational costs while maintaining or improving model performance.

### Key Contributions

- **Bandit-Guided Curriculum**: A principled approach to adaptively select data subsets during training using multi-armed bandits with submodular rewards
- **Adaptive Subset Selection**: Leverages various submodular functions (Facility Location, Graph Cut, Log Determinant) for diverse and representative subset selection that evolves with training
- **Unified Framework**: Works seamlessly across both computer vision and natural language processing tasks
- **Significant Speedups**: Achieves 2-5x training speedup with minimal accuracy loss across various benchmarks


## ✨ Key Features

- **Multi-Armed Bandit Selection**: Epsilon-greedy and UCB-based selection strategies
- **Multiple Submodular Functions**: Support for Facility Location, Graph Cut, Concave Over Modular, and Log Determinant
- **Gradient-based Selection**: Efficient gradient computation using Ghost Batch Normalization
- **Vision Tasks**: Support for CIFAR-10/100, MNIST, TinyImageNet, and more
- **LLM Fine-tuning**: Integration with Hugging Face Transformers for LLaMA-2 and other LLMs
- **LoRA Support**: Efficient parameter-efficient fine-tuning for large language models
- **Flexible Configuration**: Easy-to-use configuration system for experiments

##  Installation

### Prerequisites
- Python 3.8 or higher
- CUDA 11.8+ (for GPU support)
- PyTorch 2.0+

### Clone the Repository

```bash
git clone https://github.com/efficiency-learning/banditsubmod.git
cd banditsubmod
```

### Environment Setup

We recommend using separate environments for vision and LLM experiments.

#### For Vision Tasks

```bash
cd OnlineSubmod-vision
pip install -e .
pip install -r requirements.txt
```

#### For LLM Tasks

```bash
cd OnlineSubmod-LLM
pip install -r requirements.txt
```

## 🏃 Quick Start

> **📖 Detailed Guide**: See [QUICKSTART.md](QUICKSTART.md) for a complete 5-minute tutorial!

### Vision Tasks (CIFAR-10 Example)

```bash
cd OnlineSubmod-vision

# Install dependencies
pip install -e .
pip install -r requirements.txt

# Run example with bandit-guided curriculum
python example_cifar10.py
```

### LLM Fine-tuning (LLaMA-2 on MMLU)

```bash
cd OnlineSubmod-LLM

# Install dependencies  
pip install -r requirements.txt

# Set up Hugging Face (required for LLaMA-2)
huggingface-cli login

# Run example
bash example_mmlu.sh
```

**What you'll get:**
- **Vision**: 3x speedup, ~93.8% accuracy on CIFAR-10 (vs 94.5% full data)
- **LLM**: 2.5x speedup, ~60.8% accuracy on MMLU (vs 62.1% full data)

## 📁 Repository Structure

```
banditsubmod/
├── OnlineSubmod-vision/          # Computer vision experiments
│   ├── cords/                    # Data selection strategies
│   │   ├── selectionstrategies/  # Core selection algorithms
│   │   └── utils/                # Utility functions and models
│   ├── configs/                  # Configuration files
│   │   ├── SL/                   # Supervised learning configs
│   │   ├── SSL/                  # Semi-supervised learning configs
│   │   └── HPO/                  # Hyperparameter optimization configs
│   ├── train_online_submod_sl4_gradbatch.py  # Main training script
│   └── setup.py                  # Package installation
│
├── OnlineSubmod-LLM/             # Large language model experiments
│   ├── less/                     # LLM data selection framework
│   │   ├── data_selection/       # Data preprocessing and selection
│   │   ├── train/                # Training utilities
│   │   │   ├── gctrainer.py      # Gradient-based trainer
│   │   │   ├── submod.py         # Submodular optimization
│   │   │   └── train.py          # Main training loop
│   │   └── layers/               # Custom layers (LoRA, etc.)
│   ├── online_batch_select_mmlu.sh  # MMLU experiment script
│   └── run.sh                    # Main run script
│
├── README.md                     # This file
├── LICENSE                       # MIT License
└── CONTRIBUTING.md              # Contribution guidelines
```

## 🧪 Experiments

### Vision Experiments

Our framework supports various vision benchmarks:

#### CIFAR-10/100
```bash
# CIFAR-10 with 30% data selection
python train_online_submod_sl4_gradbatch.py \
    --config configs/SL/config_onlinesubmod_cifar10.py

# CIFAR-100 with different selection fractions
python train_online_submod_sl4_gradbatch.py \
    --config configs/SL/config_onlinesubmod_cifar100.py
```

#### MNIST
```bash
python train_online_submod_sl4_gradbatch.py \
    --config configs/SL/config_onlinesubmod_mnist.py
```

#### TinyImageNet
```bash
python train_online_submod_sl4_gradbatch.py \
    --config configs/SL/config_onlinesubmod_tinyimagenet.py
```

### LLM Experiments

#### MMLU Benchmark
```bash
cd OnlineSubmod-LLM
bash online_batch_select_mmlu.sh onlineSubmod 8 0.05 2 mmlu llama2 1 2e-05 42 1 high_school_chemistry PREFIX 2
```

#### Custom Dataset Fine-tuning
```bash
# Edit run.sh to specify your parameters
PREFIX=your_experiment_name
METHOD=onlineSubmod  # or SBERT, random, etc.
GPU=0
EVAL=your_evaluation_task
BS=8  # batch size
NVAL=2  # validation samples
FRAC=0.05  # data fraction

bash run.sh
```

### Configuration Options

Key configuration parameters for data selection:

- **Selection Strategy**: `OnlineSubmod`, `OnlineSubmodPB` (Per-Batch), `Random`, `CRAIG`, `GradMatch`, `GLISTER`
- **Submodular Function**: `facilityLocation`, `graphCut`, `logDeterminant`, `disparity-min`, `disparity-sum`
- **Selection Fraction**: Proportion of data to select (e.g., 0.3 for 30%)
- **Selection Frequency**: How often to perform selection (e.g., every epoch)
- **Bandit Parameters**: 
  - `epsilon`: Exploration rate for ε-greedy
  - `eta`: Learning rate for bandit updates
  - `lambda`: Mixing parameter for importance sampling

See `configs/README.md` for detailed configuration documentation.

## Citation

If you find this work useful, please cite our NeurIPS 2025 paper:

```bibtex
@inproceedings{chanda2025bandit,
  title={Bandit-Guided Submodular Curriculum for Adaptive Subset Selection},
  author={Chanda, Prateek and Agrawal, Prayas and Sureka, Saral and Polu, Lokesh Reddy and Kshirsagar, Atharv and Ramakrishnan, Ganesh},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/efficiency-learning/banditsubmod.git
cd banditsubmod

# Create a development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e OnlineSubmod-vision/
pip install -e OnlineSubmod-LLM/
```

## Acknowledgments

- Built on top of the [CORDS](https://github.com/decile-team/cords) framework for data subset selection
- LLM experiments adapted from [LESS](https://github.com/princeton-nlp/LESS) framework
- Submodular optimization using [submodlib](https://github.com/decile-team/submodlib)
- Thanks to the NeurIPS 2025 reviewers for their valuable feedback

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: prateek.chanda@cse.iitb.ac.in

## Star History

If you find this repository useful, please consider giving it a star ⭐!

---

**Keywords**: Data Selection, Submodular Optimization, Multi-Armed Bandits, Curriculum Learning, Efficient Deep Learning, Online Learning, Computer Vision, Natural Language Processing, LLM Fine-tuning
