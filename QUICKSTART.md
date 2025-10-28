# Quick Start Guide

This guide will help you get started with the bandit-guided submodular curriculum framework in just a few minutes.


### For Computer Vision (CIFAR-10)

```bash
# 1. Navigate to vision directory
cd OnlineSubmod-vision

# 2. Install dependencies
pip install -e .
pip install -r requirements.txt

# 3. Run the example (make sure you have a GPU)
python example_cifar10.py
```

This will train ResNet18 on CIFAR-10 with:
- 30% data selection (3x speedup)
- ~93.8% accuracy (vs 94.5% with full data)
- Adaptive curriculum learning throughout training

### For LLM Fine-tuning (LLaMA-2 on MMLU)

```bash
# 1. Navigate to LLM directory
cd OnlineSubmod-LLM

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up Hugging Face access (required for LLaMA-2)
huggingface-cli login

# 4. Run the example
bash example_mmlu.sh
```

This will fine-tune LLaMA-2-7B on MMLU with:
- 5% data selection (2.5x speedup)
- ~60.8% accuracy (vs 62.1% with full data)
- LoRA for parameter-efficient fine-tuning

##  What to Expect

### Training Output

You'll see output like:

```
Epoch: 1/300
Selection: greedy (arm 2 - Facility Location)
Selected: 15000/50000 samples (30%)
Train Loss: 1.234 | Train Acc: 56.7%
Val Loss: 1.456 | Val Acc: 54.2%
Time: 45s (vs ~150s for full data)

Epoch: 2/300
Selection: uniform (exploration)
Selected: 15000/50000 samples (30%)
...
```

### Key Metrics to Watch

1. **Selection Type**: "greedy" (exploitation) vs "uniform" (exploration)
2. **Best Arm**: Which submodular function is performing best
3. **Training Time**: Should be 2-3x faster than full data
4. **Accuracy**: Should be within 1-2% of full data training

##  Common Configuration Changes

### Change Selection Fraction

Edit the config file or set parameters:

```python
# In config file: configs/SL/config_onlinesubmod_cifar10.py
dss_args=dict(
    fraction=0.5,  # Use 50% of data instead of 30%
    ...
)
```

### Try Different Submodular Functions

The bandit automatically explores:
- Facility Location (promotes diversity)
- Graph Cut (balances similarity)
- Log Determinant (maximizes volume)

To force a specific function:

```python
dss_args=dict(
    greedy_only=True,  # Always use best arm
    submod_function="facilityLocation",  # or "graphCut", "logDeterminant"
    ...
)
```

### Adjust Exploration Rate

Control the bandit's exploration:

```python
submod_args = dict(
    pi=0.5,  # Higher = more exploration early on
    lamb=0.3,  # Mixing parameter for importance sampling
    eta_n=0.1,  # Learning rate for bandit updates
    ...
)
```

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
# In config file:
dataloader=dict(batch_size=64)  # Instead of 128
```

### Slow Selection

```bash
# Use lazy greedy optimizer
dss_args=dict(optimizer='LazyGreedy')  # Instead of 'NaiveGreedy'

# Or select less frequently
dss_args=dict(select_every=5)  # Every 5 epochs instead of 1
```

### CUDA Not Available

```bash
# Use CPU (much slower, not recommended for LLMs)
train_args=dict(device="cpu")
```

### Hugging Face Access Denied

```bash
# For LLaMA-2, you need to:
# 1. Request access at: https://huggingface.co/meta-llama/Llama-2-7b-hf
# 2. Login: huggingface-cli login
```

## Next Steps

After running the basic examples:

1. **Explore Different Datasets**
   - Vision: CIFAR-100, MNIST, TinyImageNet
   - LLM: Different MMLU subjects, custom datasets

2. **Experiment with Parameters**
   - Selection fractions (10%, 30%, 50%)
   - Bandit parameters (exploration vs exploitation)
   - Submodular function combinations

3. **Compare with Baselines**
   - Full data training
   - Random selection
   - Other methods (CRAIG, GradMatch, GLISTER)

4. **Read the Full Documentation**
   - Vision: [OnlineSubmod-vision/README.md](OnlineSubmod-vision/README.md)
   - LLM: [OnlineSubmod-LLM/README.md](OnlineSubmod-LLM/README.md)
   - Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)

## Tips for Best Results

1. **Start with Default Settings**: The default configurations are well-tuned
2. **Monitor Bandit Behavior**: Watch which arms are selected over time
3. **Use Validation Set**: A good validation set helps bandit selection

##  Getting Help

- Check [Issues](https://github.com/efficiency-learning/banditsubmod/issues)
- Read [FAQs in README](README.md)
- Contact: prateek.chanda@cse.iitb.ac.in

##  Citation

If you use this code, please cite:

```bibtex
@inproceedings{chanda2025bandit,
  title={Bandit-Guided Submodular Curriculum for Adaptive Subset Selection},
  author={Chanda, Prateek and Agrawal, Prayas and Sureka, Saral and Polu, Lokesh Reddy and Kshirsagar, Atharv and Ramakrishnan, Ganesh},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

---

