#!/bin/bash

./run.py \
    --vocab data/light_vocab.txt \
    --model-dir custom_dataset \
    --mode REPL \
    --max-sentence-length 50 \
    #--beam-width 5
