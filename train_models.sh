#! /bin/bash

PROJECT_PATH="$HOME/RL-Machine-Translation" # change this
cd $PROJECT_PATH

LEARNING_RATE=(5e-4 1e-4 3e-3)
CRITERION_LOSS=("ter" "chrf" "comet")
BATCH_SIZE=(16)
DECODER_MAX_ITER=9
MAX_EPOCHS=131

cd "$PROJECT_PATH/fairseq_extend"

for criterion in ${CRITERION_LOSS[@]}; do
    for lr in ${LEARNING_RATE[@]}; do
        for batch_size in ${BATCH_SIZE[@]}; do
            python -u train.py \
                    --config-dir "fairseq_easy_extend/models/nat/" \
                    --config-name "cmlm_config.yaml" \
                    task.data="$PROJECT_PATH/iwslt14.tokenized.de-en" \
                    dataset.batch_size=$batch_size \
                    optimization.lr=[$lr] \
                    criterion._name=$criterion \
                    checkpoint.restore_file="$PROJECT_PATH/checkpoint_best.pt" \
                    checkpoint.reset_optimizer=True \
                    checkpoint.save_dir="$PROJECT_PATH/checkpoints/${criterion}_${lr}_${batch_size}_copy" \
                    optimization.max_epoch=$MAX_EPOCHS
        done
    done
done