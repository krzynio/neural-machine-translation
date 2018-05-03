#!/bin/bash

./run.py \
    --src-vocab files/vocab.pl \
    --dst-vocab files/vocab.en \
    --model-dir luong-gru-lstm-unprocessed \
    --mode REPL \
    --max-sentence-length 39
