#!/bin/bash

./run.py \
    --src-vocab files/vocab/vocab_wiki.pl \
    --dst-vocab files/vocab/vocab_wiki.en \
    --src-train-data files/data/input_wiki_1.pl \
    --dst-train-data files/data/input_wiki_1.en \
    --src-validation-data files/data/input_wiki_1.pl \
    --dst-validation-data files/data/input_wiki_1.en \
    --src-predict-data files/data/input_wiki_1.pl \
    --dst-predict-data files/data/input_wiki_1.en \
    --model-dir luong-model-test \
    --mode TRAIN