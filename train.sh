#!/bin/bash

./run.py \
    --vocab files/vocab.pl \
    --src-train-data files/train.pl \
    --dst-train-data files/train.en \
    --src-validation-data files/test.pl \
    --dst-validation-data files/test.en \
    --src-predict-data files/valid.pl \
    --dst-predict-data files/valid.en \
    --model-dir multilang-model \
    --mode TRAIN \
    --max-sentence-length 50 \
    --batch-size 128
