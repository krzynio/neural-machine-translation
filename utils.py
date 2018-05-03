#!/usr/bin/env python3
import re

import numpy as np
import unidecode
from nltk import word_tokenize


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


def light_prepare(sentence):
    def remove_diacritics(s):
        return unidecode.unidecode(s)

    sentence = remove_diacritics(sentence)
    return word_tokenize(sentence)


def create_prepared_data(src_sentences_file,
                         dst_sentences_file,
                         prepared_src_file,
                         prepared_dst_file):
    current = 0
    with open(src_sentences_file) as source_file:
        with open(dst_sentences_file) as dst_file:
            with open(prepared_src_file, 'w') as output_src_file:
                with open(prepared_dst_file, 'w') as output_dst_file:
                    for src, dst in zip(source_file, dst_file):
                        source_sentence = ' '.join(light_prepare(src))
                        dst_sentence = ' '.join(light_prepare(dst))
                        output_src_file.write(source_sentence + '\n')
                        output_dst_file.write(dst_sentence + '\n')
                        current += 1
                        if current % 1000 == 0:
                            print(current)


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

    if len(src_lines) != len(dst_lines):
        raise Exception('src line count {} != dst line count {}'.format(len(src_lines), len(dst_lines)))
    count = len(src_lines)
    permutation = list(np.random.permutation(count))
    write_lines(src_data_out, src_lines, permutation)
    write_lines(dst_data_out, dst_lines, permutation)


def take_first_n_lines(src_in, dst_in, src_out, dst_out, n=5000000, max_length=39):
    with open(src_in) as f1_in:
        with open(dst_in) as f2_in:
            with open(src_out, 'w') as f1_out:
                with open(dst_out, 'w') as f2_out:
                    total = 0
                    while total < n:
                        s = next(f1_in).replace('\n', '').split(' ')
                        t = next(f2_in).replace('\n', '').split(' ')
                        if len(s) > max_length or len(t) > max_length:
                            continue
                        f1_out.write(' '.join(s) + '\n')
                        f2_out.write(' '.join(t) + '\n')
                        total += 1


if __name__ == '__main__':
    create_vocab('files/data5m_shuffled.en', 'files/vocab5m.en', 100000)
    create_vocab('files/data5m_shuffled.pl', 'files/vocab5m.pl', 200000)
    #shuffle_files('files/data5m.pl', 'files/data5m.en', 'files/data5m_shuffled.pl', 'files/data5m_shuffled.en')
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

    # take_first_n_lines('files/non_processed_data.pl',
    #                    'files/non_processed_data.en',
    #                    'files/data5m.pl',
    #                    'files/data5m.en',
    #                    5000000,
    #                    39)
    # create_prepared_data(
    #     'files/raw_data.pl',
    #     'files/raw_data.en',
    #     'files/non_processed_data.pl',
    #     'files/non_processed_data.en',
    #     # size=5000000,
    #     # min_length=1,
    #     # max_length=30
    # )
