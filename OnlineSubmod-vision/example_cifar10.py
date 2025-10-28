#!/usr/bin/env python
"""
Example script demonstrating bandit-guided submodular data selection on CIFAR-10.

This script shows the basic usage of our method for training a ResNet18 model
on CIFAR-10 with adaptive subset selection.

Author: Prateek Chanda et al.
Paper: "Bandit-Guided Submodular Curriculum for Adaptive Subset Selection" (NeurIPS 2025)
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cords.utils.config_utils import load_config_data
from train_sl4_gradbatch import TrainClassifier


def main():
    """
    Train ResNet18 on CIFAR-10 with bandit-guided submodular selection.
    
    This example:
    1. Loads a pre-configured setup for CIFAR-10
    2. Uses OnlineSubmod strategy with 30% data selection
    3. Trains for 300 epochs with adaptive curriculum
    4. Achieves ~3x speedup compared to full dataset training
    """
    
    print("=" * 80)
    print("Bandit-Guided Submodular Curriculum - CIFAR-10 Example")
    print("=" * 80)
    print("\nThis example demonstrates adaptive data selection on CIFAR-10")
    print("Expected speedup: ~3x | Expected accuracy: ~93.8% (vs 94.5% full data)")
    print("\nConfiguration:")
    print("  - Dataset: CIFAR-10")
    print("  - Model: ResNet18")
    print("  - Selection Strategy: OnlineSubmod")
    print("  - Selection Fraction: 30%")
    print("  - Selection Frequency: Every epoch")
    print("  - Submodular Function: Facility Location")
    print("  - Bandit: Epsilon-greedy")
    print("=" * 80)
    print()
    
    # Path to configuration file
    # You can modify this to use different configs from configs/SL/
    config_file = './configs/SL/config_onlinesubmod_cifar10.py'
    
    # Check if config exists
    if not os.path.exists(config_file):
        print(f"Error: Config file not found at {config_file}")
        print("\nAvailable example configs:")
        print("  - configs/SL/config_onlinesubmod_cifar10.py")
        print("  - configs/SL/config_onlinesubmod_cifar100.py")
        print("  - configs/SL/config_onlinesubmod_mnist.py")
        print("\nPlease create a config file or specify an existing one.")
        return
    
    # Load configuration
    print(f"Loading configuration from: {config_file}")
    cfg = load_config_data(config_file)
    
    # Display key parameters
    print("\nKey Parameters:")
    print(f"  Selection Type: {cfg.dss_args.get('type', 'N/A')}")
    print(f"  Selection Fraction: {cfg.dss_args.get('fraction', 'N/A')}")
    print(f"  Submodular Function: {cfg.dss_args.get('submod_function', 'N/A')}")
    print(f"  Batch Size: {cfg.dataloader.get('batch_size', 'N/A')}")
    print(f"  Number of Epochs: {cfg.train_args.get('num_epochs', 'N/A')}")
    print(f"  Learning Rate: {cfg.optimizer.get('lr', 'N/A')}")
    print()
    
    # Initialize trainer
    print("Initializing trainer...")
    clf = TrainClassifier(cfg)
    
    # Start training
    print("\nStarting training with bandit-guided curriculum...")
    print("-" * 80)
    clf.train()
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
