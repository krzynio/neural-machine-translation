#!/usr/bin/env python3
import re

import numpy as np
import unidecode


def load_vocab(path):
    with open(path) as f:
        words = f.read().split('\n')[:-1]
        return dict(map(lambda x: (x[1], x[0]), list(enumerate(words)))), dict(
            map(lambda x: (x[0], x[1]), enumerate(words)))


def prepare_sentence(sentence):
    def remove_repeating_spaces(sentence):
        return re.sub('[ ]{2,}', ' ', sentence)

    def remove_non_alphanumeric(sentence):
        return re.sub('[^0-9a-zA-Z ]+', '', sentence)

    def remove_diacritics(sentence):
        return unidecode.unidecode(sentence)

    def tokenize(sentence):
        return sentence.split(' ')

    return remove_repeating_spaces(remove_non_alphanumeric(remove_diacritics(sentence))).strip().lower()


def create_vocab(data_file,
                 output_file,
                 vocab_size,
                 start_token='<s>',
                 end_token='</s>',
                 unknown_token='</unk>'):
    words = {}
    with open(data_file) as f:
        for line in f:
            line = line.replace('\n', '')
            tokens = line.split(' ')
            for word in tokens:
                if word in words:
                    words[word] = 1 + words[word]
                else:
                    words[word] = 1
    words = [end_token, start_token, unknown_token] + list(
        map(lambda x: x[0], sorted(words.items(), key=lambda x: x[1], reverse=True)))[:vocab_size - 3]
    with open(output_file, 'w') as f:
        for w in words:
            f.write(w + '\n')


def create_prepared_data(src_sentences_file,
                         dst_sentences_file,
                         prepared_src_file,
                         prepared_dst_file,
                         size=50000,
                         min_length=3,
                         max_length=30):
    with open(src_sentences_file) as source_file:
        with open(dst_sentences_file) as dst_file:
            with open(prepared_src_file, 'w') as output_src_file:
                with open(prepared_dst_file, 'w') as output_dst_file:
                    total = 0
                    while total < size:
                        source_sentence = next(source_file)
                        dst_sentence = next(dst_file)

                        source_sentence = prepare_sentence(source_sentence)
                        dst_sentence = prepare_sentence(dst_sentence)
                        src_len = len(source_sentence.split(' '))
                        dst_len = len(dst_sentence.split(' '))

                        if src_len > max_length or src_len < min_length or dst_len > max_length or dst_len < min_length:
                            continue
                        total += 1
                        output_src_file.write(source_sentence + '\n')
                        output_dst_file.write(dst_sentence + '\n')


def read_lines(file):
    with open(file) as f:
        return f.read().split('\n')[:-1]


def write_lines(file, lines, permutation):
    with open(file, 'w') as f:
        for index in permutation:
            f.write(lines[index] + '\n')


def shuffle_files(src_data_in, dst_data_in, src_data_out, dst_data_out):
    src_lines = read_lines(src_data_in)
    dst_lines = read_lines(dst_data_in)

    if len(src_lines) != dst_lines:
        raise Exception('src line count {} != dst line count {}'.format(len(src_lines), len(dst_lines)))
    count = len(src_lines)
    permutation = list(np.random.permutation(count))
    write_lines(src_data_out, src_lines, permutation)
    write_lines(dst_data_out, dst_lines, permutation)


if __name__ == '__main__':
    # create_vocab('files/data_shuffled.en', 'files/vocab.en', 50000)
    create_vocab('files/data_shuffled.pl', 'files/vocab.pl', 70000)
    # shuffle_files('files/data.pl', 'files/data.en', 'files/data_shuffled.pl', 'files/data_shuffled.en')
    # count_pl = 0
    # count_en = 0
    # with open('files/data.pl', 'r') as f:
    #     for line in f:
    #         if len(line.replace('\n', '').split(' ')) > 30:
    #             count_pl += 1
    # with open('files/data.en', 'r') as f:
    #     for line in f:
    #         if len(line.replace('\n', '').split(' ')) > 30:
    #             count_en += 1
    # print(count_pl, count_en)

    # create_prepared_data(
    #     'files/raw_data.pl',
    #     'files/raw_data.en',
    #     'files/data.pl',
    #     'files/data.en',
    #     size=5000000,
    #     min_length=1,
    #     max_length=30
    # )
