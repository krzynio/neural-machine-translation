#!/bin/bash

./nmt.py \
    --src-vocab files/vocab/vocab_wiki.pl \
    --dst-vocab files/vocab/vocab_wiki.en \
    --src-train-data files/data/input_wiki.pl \
    --dst-train-data files/data/input_wiki.en \
    --model-location files/models/enc-dec-attn \
    --batch-size 1
