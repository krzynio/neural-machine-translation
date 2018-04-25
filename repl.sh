#!/bin/bash

./run.py \
    --src-vocab files/vocab/vocab_wiki.pl \
    --dst-vocab files/vocab/vocab_wiki.en \
    --model-dir luong-model-test \
    --mode REPL