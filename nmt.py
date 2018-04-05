#!/usr/bin/env python3

import argparse

from data.data_generator import DataGenerator
from data.sentence_encoding import SentenceEncoding
from model import NMTModel
from vocab.vocab import Vocabulary


def parse_arguments():
    parser = argparse.ArgumentParser(description='Neural machine translation')
    parser.add_argument('--src-vocab', type=str, help='location of src vocabulary', required=True)
    parser.add_argument('--dst-vocab', type=str, help='location of dst vocabulary', required=True)

    parser.add_argument('--model-location', type=str, help='location of the model', required=True)
    parser.add_argument('--mode', type=str, help='program mode', default='TRAIN', choices=['TRAIN', 'REPL'])

    parser.add_argument('--src-train-data', type=str, help='location of src train data file')
    parser.add_argument('--dst-train-data', type=str, help='location of dst train data file')

    parser.add_argument('--src-validation-data', type=str, help='location of src validation data file')
    parser.add_argument('--dst-validation-data', type=str, help='location of dst validation data file')

    parser.add_argument('--lstm-units', type=int, default=1024, help='number of lstm units')
    parser.add_argument('--embedding-size', type=int, default=300, help='embedding size')
    parser.add_argument('--max-sentence-length', type=int, default=45, help='max sentence length')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')

    args = parser.parse_args()

    if args.mode == 'TRAIN':
        if args.src_train_data is None or args.dst_train_data is None:
            parser.error('src-train-data and dst-train-data are required when mode = TRAIN')
        if args.src_validation_data is None and args.dst_validation_data is not None or args.src_validation_data is not None and args.dst_validation_data is None:
            parser.error(
                'src-validation-data and dst-validation-data must both be either None or not None when mode = TRAIN')

    return args


def main(args):
    src_vocab = Vocabulary.read_from_file(args.src_vocab)
    dst_vocab = Vocabulary.read_from_file(args.dst_vocab)

    args.src_vocab_size = src_vocab.size
    args.start_token = dst_vocab.encode(dst_vocab.start_token)
    args.end_token = dst_vocab.encode(dst_vocab.start_token)
    args.dst_vocab_size = dst_vocab.size

    src_encoding = SentenceEncoding(src_vocab)
    dst_encoding = SentenceEncoding(dst_vocab)

    model = NMTModel(args, src_encoding, dst_encoding)

    if args.mode == 'REPL':
        while 1:
            sentence = input('>> ')
            print('Input: {}'.format(sentence))
            decoded = model.predict([sentence])
            print('Translation: {}'.format(decoded[0]))
    else:
        data_generator = DataGenerator(args.src_train_data, args.dst_train_data)
        model.train(data_generator)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
