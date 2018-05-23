#!/bin/bash

./run.py \
    --src-vocab files/vocab.pl \
    --dst-vocab files/vocab.en \
    --model-dir luong-gru-lstm-unprocessed \
    --mode BLEU \
    --src-validation-data files/test.pl --dst-validation-data files/test.en \
    --max-sentence-length 39
