#!/bin/bash

./run.py \
    --src-vocab files/vocab.pl \
    --dst-vocab files/vocab.en \
    --src-train-data files/train.pl \
    --dst-train-data files/train.en \
    --src-validation-data files/test.pl \
    --dst-validation-data files/test.en \
    --src-predict-data files/valid.pl \
    --dst-predict-data files/valid.en \
    --model-dir luong-gru-lstm-unprocessed \
    --mode TRAIN \
    --max-sentence-length 39
