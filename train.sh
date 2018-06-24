#!/bin/bash

./run.py \
    --vocab data/light_vocab.txt \
    --src-train-data data/small_src.txt \
    --dst-train-data data/small_dst.txt \
    --src-valid-data data/small_src.txt \
    --dst-valid-data data/small_dst.txt \
    --model-dir custom_dataset \
    --mode TRAIN \
    --max-sentence-length 50 \
    --batch-size 2 \
    --epochs-count 1