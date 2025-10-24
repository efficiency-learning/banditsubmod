PREFIX=2shot_SBERT_chem
METHOD=SBERT
GPU=6                               #DONT USE MULTIPLE
EVAL=high_school_chemistry
BS=8
NVAL=2
FRAC=0.05


export HF_HOME="$HF_DIR/.hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_METRICS_CACHE="$HF_HOME/metrics"

screen -dmS $PREFIX bash -c "
  CUDA_VISIBLE_DEVICES=$GPU \
  sh online_batch_select_mmlu.sh \
    $METHOD \
    $BS \
    $FRAC \
    $NVAL \
    mmlu \
    llama2 \
    1 \
    2e-05 \
    42 \
    1 \
    $EVAL \
    $PREFIX \
    2 > ./logs/logs_$PREFIX.log 2>&1;
  exit
"
