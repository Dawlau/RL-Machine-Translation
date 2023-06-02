#! /bin/bash

CRITERIONS=("chrf" "ter" "comet")
LEARNING_RATES=(1e-4 3e-3 5e-4)

PROJECT_PATH="$HOME/RL-Machine-Translation"

cd $PROJECT_PATH
cd fairseq_extend

for criterion in ${CRITERIONS[@]}; do
    for lr in ${LEARNING_RATES[@]}; do
        checkpoint="$PROJECT_PATH/checkpoints/${criterion}_${lr}_16/checkpoint_best.pt"

        python decode.py "$PROJECT_PATH/iwslt14.tokenized.de-en" --source-lang de --target-lang en \
            --path $checkpoint \
            --task translation_lev \
            --iter-decode-max-iter 9 \
            --gen-subset test \
            --print-step \
            --remove-bpe \
            --tokenizer moses \
            --scoring bleu
        break
    done
    break
done