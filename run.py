#!/usr/bin/env python3

import argparse
import logging

import tensorflow as tf

from model import TranslatorModel
from utils import setup_logger
from dataset import DataSet


def parse_arguments():
    parser = argparse.ArgumentParser(description='Neural machine translation')
    parser.add_argument('--vocab', type=str, help='location of vocabulary', required=True)

    parser.add_argument('--model-dir', type=str, help='location of the model', required=True)
    parser.add_argument('--mode', type=str, help='program mode', default='TRAIN', choices=['TRAIN', 'REPL', 'BLEU'])

    parser.add_argument('--src-train-data', type=str, help='location of src train data file')
    parser.add_argument('--dst-train-data', type=str, help='location of dst train data file')

    parser.add_argument('--log-file', type=str, help='path to log file', default='nmt.log')

    parser.add_argument('--src-valid-data', type=str, help='location of src validation data file')
    parser.add_argument('--dst-valid-data', type=str, help='location of dst validation data file')

    parser.add_argument('--cell-units', type=int, default=512, help='number of cell units')
    parser.add_argument('--embedding-size', type=int, default=300, help='embedding size')
    parser.add_argument('--max-sentence-length', type=int, default=50, help='max sentence length')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs-count', type=int, default=10, help='count of epochs')
    parser.add_argument('--beam-width', type=int, default=None, help='decoder beam width (disabled by default)')

    args = parser.parse_args()

    if args.mode == 'TRAIN':
        if args.src_train_data is None or args.dst_train_data is None:
            parser.error('src-train-data and dst-train-data are required when mode = TRAIN')

    if args.mode == 'BLEU':
        if args.src_valid_data is None or args.dst_valid_data is None:
            parser.error('src-valid-data and dst-valid-data are required when mode = BLEU')

    if args.beam_width == 0:
        args.beam_width = None

    return args


def train(args, translator):
    train_ds = DataSet(args.src_train_data, args.dst_train_data, translator.vocab)
    valid_ds = None
    if args.src_valid_data is not None:
        valid_ds = DataSet(args.src_valid_data, args.dst_valid_data, translator.vocab)
    translator.train(train_dataset=train_ds,
                     epochs=args.epochs_count,
                     batch_size=args.batch_size,
                     validation_dataset=valid_ds,
                     predict_samples=100)


def bleu(args, translator):
    ds = DataSet(args.src_valid_data, args.dst_valid_data, translator.vocab)
    generator, _, references = ds.new_generator(batch_size=args.batch_size,
                                                max_length=args.max_sentence_length,
                                                limit=100,
                                                return_raw=True)
    references = [r.split(' ') for r in references]
    bleu = translator.calculate_bleu(generator, references)
    print('BLEU = {}'.format(bleu[0]))


def repl(args, translator):
    from utils import prepare
    try:
        while 1:
            sentence = ' '.join(prepare(input('>> ')))
            generator = DataSet.generator_from_list([sentence], translator.vocab, args.batch_size, args.max_sentence_length)
            for translation in translator.translate(generator):
                print(translation)
    except Exception as e:
        print(e)
        pass


def main():
    tf.logging._logger.setLevel(logging.INFO)
    config = tf.estimator.RunConfig(
        save_summary_steps=100,
        session_config=None,
        keep_checkpoint_max=1,
        keep_checkpoint_every_n_hours=10000,
        log_step_count_steps=100
    )
    args = parse_arguments()
    setup_logger(args.log_file)
    translator = TranslatorModel(args, config)

    if args.mode == 'TRAIN':
        train(args, translator)
    elif args.mode == 'BLEU':
        bleu(args, translator)
    else:
        repl(args, translator)


if __name__ == '__main__':
    main()
