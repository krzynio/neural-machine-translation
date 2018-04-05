#!/bin/bash

./nmt.py \
    --src-vocab files/vocab/vocab_wiki.pl \
    --dst-vocab files/vocab/vocab_wiki.en \
    --model-location files/models/enc-dec-attn \
    --mode REPL
