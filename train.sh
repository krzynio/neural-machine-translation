#!/bin/bash

./run.py \
    --vocab data/lvocab.txt \
    --src-train-data data/train.src \
    --dst-train-data data/train.dst \
    --src-valid-data data/test.src \
    --dst-valid-data data/test.dst \
    --model-dir models/multilingual_vocab100k_embed128 \
    --mode TRAIN \
    --max-sentence-length 50 \
    --batch-size 128 \
    --embedding-size 128 \
    --epochs-count 10
