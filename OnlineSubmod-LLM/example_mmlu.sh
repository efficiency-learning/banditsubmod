#!/bin/bash
#
# Example script for fine-tuning LLaMA-2 on MMLU with bandit-guided submodular selection
#
# Author: Prateek Chanda et al.
# Paper: "Bandit-Guided Submodular Curriculum for Adaptive Subset Selection" (NeurIPS 2025)
#
# This script demonstrates basic usage of our method for LLM fine-tuning.
# It trains LLaMA-2-7B on a subset of the MMLU benchmark using LoRA and
# adaptive data selection with bandit-guided curriculum learning.
#

set -e  # Exit on error

echo "================================================================================"
echo "Bandit-Guided Submodular Curriculum - LLaMA-2 MMLU Example"
echo "================================================================================"
echo ""
echo "This example demonstrates adaptive data selection for LLM fine-tuning"
echo "Expected speedup: ~2.5x | Expected accuracy: ~60.8% (vs 62.1% full data)"
echo ""
echo "Configuration:"
echo "  - Model: LLaMA-2-7B"
echo "  - Task: MMLU (high_school_chemistry)"
echo "  - Selection Strategy: onlineSubmod"
echo "  - Selection Fraction: 5%"
echo "  - LoRA Rank: 8, Alpha: 1"
echo "  - Batch Size: 8"
echo "================================================================================"
echo ""

# Configuration parameters
PREFIX="mmlu_example"
METHOD="onlineSubmod"  # Options: onlineSubmod, SBERT, random
GPU=0                  # GPU device ID
EVAL="high_school_chemistry"  # MMLU subject to evaluate on
BS=8                   # Batch size
NVAL=2                 # Number of validation samples
FRAC=0.05              # Data selection fraction (5%)
EVAL_BS=2              # Evaluation batch size

# Set up environment variables for Hugging Face cache
export HF_HOME="${HOME}/.hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_METRICS_CACHE="${HF_HOME}/metrics"

echo "Environment Setup:"
echo "  HF_HOME: ${HF_HOME}"
echo "  GPU: ${GPU}"
echo ""

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. Make sure CUDA is properly installed."
    echo "You can still run on CPU but it will be very slow."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if the script exists
if [ ! -f "online_batch_select_mmlu.sh" ]; then
    echo "Error: online_batch_select_mmlu.sh not found!"
    echo "Please make sure you're running this from the OnlineSubmod-LLM directory."
    exit 1
fi

echo "Starting fine-tuning with bandit-guided curriculum..."
echo "--------------------------------------------------------------------------------"
echo ""

# Run the training script
CUDA_VISIBLE_DEVICES=${GPU} \
bash online_batch_select_mmlu.sh \
    ${METHOD} \
    ${BS} \
    ${FRAC} \
    ${NVAL} \
    mmlu \
    llama2 \
    1 \
    2e-05 \
    42 \
    1 \
    ${EVAL} \
    ${PREFIX} \
    ${EVAL_BS}

echo ""
echo "================================================================================"
echo "Training completed!"
echo "================================================================================"
echo ""
echo "Results saved with prefix: ${PREFIX}"
echo ""
echo "Key Features Demonstrated:"
echo "  ✓ Bandit-guided selection among multiple submodular functions"
echo "  ✓ Adaptive curriculum that evolves during training"
echo "  ✓ Efficient LoRA fine-tuning for large language models"
echo "  ✓ Significant speedup with minimal accuracy loss"
echo ""
echo "To experiment further:"
echo "  - Try different MMLU subjects (change EVAL variable)"
echo "  - Adjust selection fraction (FRAC=0.1 for 10%)"
echo "  - Compare with baselines (METHOD=random or METHOD=SBERT)"
echo "  - Modify bandit parameters in less/train/gctrainer.py"
echo ""
echo "For more information, see the README.md file."
echo "================================================================================"
