#!/bin/bash

./run.py \
    --vocab data/light_vocab.txt \
    --model-dir custom_dataset \
    --mode BLEU \
    --src-valid-data data/small_src.txt \
    --dst-valid-data data/small_dst.txt \
    --max-sentence-length 50 \
    --beam-width 5
