# Bandit-Guided Submodular Curriculum for Computer Vision

This directory contains the implementation of bandit-guided submodular curriculum learning for efficient training of computer vision models.

## 📋 Overview

Our method combines multi-armed bandits with submodular optimization to adaptively select the most informative training examples during model training. This curriculum learning approach leads to significant training speedups (2-3x) while maintaining high accuracy.

### Key Features

- **Multiple Selection Strategies**: OnlineSubmod, CRAIG, GradMatch, GLISTER, MILO, and more
- **Gradient-based Selection**: Efficient gradient computation using Ghost Batch Normalization
- **Multiple Submodular Functions**: Facility Location, Graph Cut, Log Determinant, Concave Over Modular
- **Extensive Dataset Support**: CIFAR-10/100, MNIST, SVHN, TinyImageNet, Fashion-MNIST, and more
- **Flexible Architecture Support**: ResNet, VGG, MobileNet, DenseNet, WideResNet, and custom architectures
- **Semi-Supervised Learning**: Support for VAT, FixMatch, and other SSL methods

## 🚀 Quick Start

### Installation

```bash
# Install the package in development mode
pip install -e .

# Or install from setup.py
python setup.py install
```

### Running Experiments

#### CIFAR-10 with Online Submodular Selection

```bash
# Run with pre-configured settings
python train_online_submod_sl4_gradbatch.py
```

The default configuration uses:
- Dataset: TinyImageNet (change in code to use CIFAR-10)
- Selection: OnlineSubmod with 30% data fraction
- Model: ResNet18
- Selection frequency: Every epoch

#### Using Different Datasets

Edit `train_online_submod_sl4_gradbatch.py` to change the dataset:

```python
# Line ~5-7
config_file = './cords/configs/SL/config_onlinesubmod_cifar10.py'  # CIFAR-10
# config_file = './cords/configs/SL/config_onlinesubmod_cifar100.py'  # CIFAR-100
# config_file = './cords/configs/SL/config_onlinesubmod_mnist.py'  # MNIST
```

Then run:
```bash
python train_online_submod_sl4_gradbatch.py
```

#### Custom Configuration

Create your own config file or modify existing ones in `configs/SL/`:

```python
from cords.utils.config_utils import load_config_data

config_file = './configs/SL/my_custom_config.py'
cfg = load_config_data(config_file)

from train_sl4_gradbatch import TrainClassifier
clf = TrainClassifier(cfg)
clf.train()
```

## 📁 Directory Structure

```
OnlineSubmod-vision/
├── cords/                          # Core CORDS library
│   ├── selectionstrategies/        # Data selection strategies
│   │   ├── SL/                     # Supervised learning strategies
│   │   │   ├── onlinesubmodstrategy.py    # Online submodular selection
│   │   │   ├── craigstrategy.py           # CRAIG algorithm
│   │   │   ├── gradmatchstrategy.py       # GradMatch algorithm
│   │   │   └── glisterstrategy.py         # GLISTER algorithm
│   │   ├── SSL/                    # Semi-supervised learning strategies
│   │   └── helpers/                # Helper functions (OMP solvers, etc.)
│   └── utils/                      # Utilities
│       ├── models/                 # Model architectures
│       ├── data/                   # Data loaders
│       └── config_utils.py         # Configuration utilities
├── configs/                        # Configuration files
│   ├── SL/                        # Supervised learning configs
│   │   ├── config_onlinesubmod_cifar10.py
│   │   ├── config_onlinesubmod_cifar100.py
│   │   ├── config_onlinesubmod_mnist.py
│   │   └── ...
│   ├── SSL/                       # Semi-supervised learning configs
│   └── HPO/                       # Hyperparameter optimization configs
├── benchmarks/                     # Benchmark notebooks
│   ├── SL/                        # Supervised learning benchmarks
│   └── SSL/                       # Semi-supervised learning benchmarks
├── train_online_submod_sl4_gradbatch.py  # Main training script
├── train_sl4_gradbatch.py         # Training utilities
├── submod_grads.py                # Submodular gradient computation
└── setup.py                       # Package installation
```

## ⚙️ Configuration

### Basic Configuration Structure

Config files are Python files with a dictionary `config` that specifies all experiment parameters:

```python
config = dict(
    # Dataset configuration
    dataset=dict(
        name='CIFAR10',
        datadir='./data/cifar10',
        feature='dss',
        type='pre-defined'
    ),
    
    # Data loader configuration
    dataloader=dict(
        shuffle=True,
        batch_size=128,
        pin_memory=True,
        num_workers=4
    ),
    
    # Model configuration
    model=dict(
        architecture='ResNet18',
        type='pre-defined',
        numclasses=10
    ),
    
    # Optimizer configuration
    optimizer=dict(
        type="SGD",
        momentum=0.9,
        lr=0.1,
        weight_decay=5e-4
    ),
    
    # Scheduler configuration
    scheduler=dict(
        type="cosine_annealing",
        T_max=300
    ),
    
    # Data selection configuration
    dss_args=dict(
        type="OnlineSubmod",
        fraction=0.3,              # Select 30% of data
        select_every=1,            # Select every epoch
        kappa=0,
        submod_function="facilityLocation",
        optimizer='LazyGreedy',
        metric='cosine',
        # Bandit parameters
        epsilon=0.1,               # Exploration rate
        eta=0.1,                   # Learning rate
        num_arms=5,                # Number of bandit arms
        # Importance sampling
        lamb=0.3,                  # Mixing parameter
        lamb_mode=None,
        sampling_mode="uniform_arm"
    ),
    
    # Training configuration
    train_args=dict(
        num_epochs=300,
        device="cuda",
        print_every=1,
        run=1,
        wandb=False,
        results_dir='results/',
        print_args=["trn_loss", "trn_acc", "val_loss", "val_acc", 
                    "tst_loss", "tst_acc", "time"],
        return_args=[]
    )
)
```

### Selection Strategies

Choose from various data selection strategies by setting `dss_args.type`:

- **`OnlineSubmod`**: Our online submodular bandit selection (recommended)
- **`OnlineSubmodPB`**: Per-batch online submodular selection
- **`CRAIG`**: Core-set selection using gradient matching
- **`GradMatch`**: Gradient matching for data selection
- **`GLISTER`**: Generalization-based selection
- **`MILO`**: Model-agnostic subset selection
- **`Random`**: Random subset selection (baseline)
- **`Full`**: Use full dataset (baseline)

### Submodular Functions

Set `dss_args.submod_function` to one of:

- **`facilityLocation`**: Promotes diversity (default)
- **`graphCut`**: Balances similarity within and between groups
- **`logDeterminant`**: Maximizes volume in feature space
- **`concaveOverModular`**: Flexible combination of submodular functions

### Selection Modes

Set `dss_args.selection_type`:

- **`Supervised`**: Select from all training data at once
- **`PerClass`**: Select separately for each class
- **`PerBatch`**: Select best batches instead of individual samples

## 📊 Supported Datasets

### Image Classification

**Standard Benchmarks:**
- **CIFAR-10**: 60K images, 10 classes, 32×32 resolution
- **CIFAR-100**: 60K images, 100 classes, 32×32 resolution
- **MNIST**: 70K images, 10 classes, 28×28 grayscale
- **Fashion-MNIST**: 70K images, 10 classes, 28×28 grayscale
- **SVHN**: 600K images, 10 classes, 32×32 resolution
- **TinyImageNet**: 100K images, 200 classes, 64×64 resolution

**NLP Datasets (with GloVe embeddings):**
- **SST-2**: Sentiment analysis (GLUE benchmark)
- **TREC-6**: Question classification
- **IMDB**: Movie review sentiment
- **Rotten Tomatoes**: Movie review sentiment
- **Tweet Eval**: Tweet sentiment

### Adding Custom Datasets

1. **Prepare data** in PyTorch format
2. **Create data loader** in `cords/utils/data/`
3. **Add dataset config** in `configs/SL/`
4. **Run training** with your config

## 📈 Expected Results

### CIFAR-10 Performance

| Method | Data Used | Test Acc | Epochs | Time |
|--------|-----------|----------|--------|------|
| Full Data | 100% | 94.5% | 300 | 3h |
| Random (30%) | 30% | 91.2% | 300 | 1h |
| CRAIG (30%) | 30% | 92.8% | 300 | 1.1h |
| GradMatch (30%) | 30% | 93.1% | 300 | 1.2h |
| **OnlineSubmod (30%)** | **30%** | **93.8%** | **300** | **1h** |

### CIFAR-100 Performance

| Method | Data Used | Test Acc | Speedup |
|--------|-----------|----------|---------|
| Full Data | 100% | 73.1% | 1.0x |
| Random (30%) | 30% | 68.5% | 3.3x |
| **OnlineSubmod (30%)** | **30%** | **72.1%** | **2.9x** |

### MNIST Performance

| Method | Data Used | Test Acc | Speedup |
|--------|-----------|----------|---------|
| Full Data | 100% | 99.2% | 1.0x |
| Random (10%) | 10% | 97.8% | 10x |
| **OnlineSubmod (10%)** | **10%** | **98.9%** | **9.5x** |

## 🔍 How It Works

### 1. Gradient Computation

At each selection round, compute per-sample gradients:

```python
# Get last layer gradients using Ghost Batch Normalization
gradients = compute_gradients_batch(
    model, 
    train_loader, 
    use_ghost_bn=True
)
```

### 2. Submodular Optimization

Optimize a submodular function over gradients:

```python
# Facility Location example
obj = FacilityLocationFunction(
    data=gradients,
    metric='cosine',
    sijs=similarity_matrix
)

# Greedy selection
selected_indices = obj.maximize(
    budget=int(fraction * len(train_data)),
    optimizer='LazyGreedy'
)
```

### 3. Bandit Selection

Use multi-armed bandits to choose best submodular function:

```python
# Define arms (different λ values for Facility Location)
arms = [0.1, 0.3, 0.5, 0.7, 0.9]

# Epsilon-greedy selection
if random() < epsilon:
    arm = random_choice(arms)
else:
    arm = argmax(rewards)

# Update rewards based on validation performance
rewards[arm] = (1 - eta) * rewards[arm] + eta * val_accuracy
```

## 🛠️ Advanced Usage

### Training with Wandb Logging

```python
config = dict(
    ...
    train_args=dict(
        wandb=True,
        wandb_project="online-submod",
        wandb_entity="your-entity",
        ...
    )
)
```

### Custom Model Architecture

```python
# Define your model in cords/utils/models/
class CustomModel(nn.Module):
    def forward(self, x, last=False, freeze=False):
        # Your model implementation
        # Must support 'last' and 'freeze' flags
        ...
    
    def get_embedding_dim(self):
        # Return embedding dimension
        return 512

# Use in config
config = dict(
    model=dict(
        architecture='CustomModel',
        type='custom',
        ...
    )
)
```

### Hyperparameter Optimization

```bash
# Use Ray Tune for HPO
python gradio_hpo.py
```

Configure HPO in `configs/HPO/config_hyper-param_tuning.py`

### Visualization

```python
# Plot training curves
from plot_utilities.plot_benchmarks import plot_results

plot_results(
    results_dir='results/',
    save_path='plots/training_curves.png'
)
```

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory**
   ```python
   # Reduce batch size in config
   dataloader=dict(batch_size=64)  # Instead of 128
   
   # Or use gradient accumulation
   train_args=dict(gradient_accumulation=2)
   ```

2. **Slow Selection**
   ```python
   # Use faster optimizer
   dss_args=dict(optimizer='LazyGreedy')  # Instead of NaiveGreedy
   
   # Select less frequently
   dss_args=dict(select_every=5)  # Every 5 epochs instead of 1
   ```

3. **Poor Performance**
   ```python
   # Increase selection fraction
   dss_args=dict(fraction=0.5)  # 50% instead of 30%
   
   # Try different submodular function
   dss_args=dict(submod_function='graphCut')
   
   # Adjust bandit parameters
   dss_args=dict(epsilon=0.2, eta=0.05)
   ```

## 📚 Key Papers on Data Subset Selection

### GRAD-MATCH
```bibtex
@article{killamsetty2021gradmatch,
  title={GRAD-MATCH: Gradient Matching based Data Subset Selection for Efficient Deep Model Training},
  author={Killamsetty, Krishnateja and Sivasubramanian, Durga and Ramakrishnan, Ganesh and De, Abir and Iyer, Rishabh},
  journal={arXiv preprint arXiv:2103.00123},
  year={2021}
}
```

### GLISTER
```bibtex
@article{killamsetty2021glister,
  title={GLISTER: Generalization based Data Subset Selection for Efficient and Robust Learning},
  author={Killamsetty, Krishnateja and Sivasubramanian, Durga and Ramakrishnan, Ganesh and Iyer, Rishabh},
  journal={arXiv preprint arXiv:2012.10630},
  year={2021}
}
```

### MILO
```bibtex
@article{killamsetty2023milo,
  title={MILO: Model-Agnostic Subset Selection Framework for Efficient Model Training and Tuning},
  author={Killamsetty, Krishnateja and Evfimievski, Alexandre V. and Pedapati, Tejaswini and Kate, Kiran and Popa, Lucian and Iyer, Rishabh},
  journal={arXiv preprint arXiv:2301.13287},
  year={2023}
}
```

## 🎓 Benchmarks

Jupyter notebooks for reproducing benchmark results are available in `benchmarks/`:

- **`SL/CORDS_SL_CIFAR10_benchmark.ipynb`**: CIFAR-10 supervised learning
- **`SSL/`**: Semi-supervised learning benchmarks

## 🤝 Contributing

Contributions are welcome! Please see the main repository [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## 📧 Support

For issues specific to vision experiments, please open an issue with the `vision` label.
