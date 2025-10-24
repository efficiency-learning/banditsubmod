PREFIX=tydiqa_tydiqa_GradNorm_bs8_seq_256
METHOD=GradNorm
GPU=0                              #DONT USE MULTIPLE
EVAL=DUMMY
BS=8
NVAL=1

screen -dmS $PREFIX bash -c "
  CUDA_VISIBLE_DEVICES=$GPU \
  sh online_batch_select_mmlu_mistral.sh \
    $METHOD \
    $BS \
    0.4 \
    $NVAL \
    tydiqa \
    mistral7b \
    1 \
    1e-05 \
    42 \
    1 \
    $EVAL \
    $PREFIX \
    2 > ./logs/logs_$PREFIX.log 2>&1;
  exit
"
