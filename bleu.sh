#!/bin/bash

./run.py \
    --vocab data/lvocab.txt \
    --model-dir models/multilingual_vocab100k_embed128 \
    --mode BLEU \
    --src-valid-data data/zeroshot5k.src \
    --dst-valid-data data/zeroshot5k.dst \
    --max-sentence-length 50 \
    --batch-size 32 \
    --embedding-size 128 #\
    #--beam-width 10
