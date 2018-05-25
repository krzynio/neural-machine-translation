#!/bin/bash

./run.py \
    --src-vocab files/vocab.pl \
    --dst-vocab files/vocab.en \
    --model-dir luong-gru-lstm-unprocessed \
    --mode BLEU \
    --src-validation-data $1 --dst-validation-data $2 \
    --max-sentence-length 39
