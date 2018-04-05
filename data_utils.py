#!/usr/bin/env python3
import re

import numpy as np
import unidecode


class Vocabulary:

    def __init__(self, words):
        self._word_to_int = dict([(w, i) for i, w in enumerate(words)])
        self._int_to_word = dict([(i, w) for i, w in enumerate(words)])
        self.words = words
        self.unknown_token = words[0]
        self.start_token = words[1]
        self.end_token = words[2]
        self.size = len(words)

    def word_to_int(self, word):
        if word in self._word_to_int:
            return self._word_to_int[word]
        else:
            return self._word_to_int[self.unknown_token]

    def int_to_word(self, val):
        if val < self.size:
            return self._int_to_word[val]
        else:
            raise Exception('word given by index {} not found'.format(val))

    def write_to_file(self, filename):
        with open(filename, 'w') as f:
            for w in self.words:
                print(w)
                f.write(w + '\n')

    @staticmethod
    def read_from_file(filename):
        words = []
        with open(filename) as f:
            for line in f:
                word = line.replace('\n', '')
                words.append(word)
        return Vocabulary(words)

    @staticmethod
    def create_from_file(filename, vocab_size=2000, start_token='<s>', end_token='</s>', unknown_token='<unk>'):
        words = {}
        with open(filename) as f:
            for line in f:
                line = line.replace('\n', '')
                tokens = line.split(' ')
                for word in tokens:
                    if word in words:
                        words[word] = 1 + words[word]
                    else:
                        words[word] = 1
        return Vocabulary([unknown_token, start_token, end_token] + list(
            map(lambda x: x[0], sorted(words.items(), key=lambda x: x[1], reverse=True)))[:vocab_size - 3])


class DataGenerator:

    def __init__(self, src_vocab_filename, src_data_filename, dst_vocab_filename, dst_data_filename):
        self.src_vocab = Vocabulary.read_from_file(src_vocab_filename)
        self.dst_vocab = Vocabulary.read_from_file(dst_vocab_filename)
        self.src_data_filename = src_data_filename
        self.dst_data_filename = dst_data_filename
        self.src_file, self.dst_file = self.open_files()

    def open_files(self):
        return open(self.src_data_filename), open(self.dst_data_filename)

    def next_batch(self, size, padding):
        total = 0
        inputs = []
        outputs = []
        #targets = []
        labels = []
        while total < size:
            try:
                src = next(self.src_file)
                dst = next(self.dst_file)
                src = self._prepare_sentence(src, self.src_vocab, padding)
                dst = self._prepare_sentence(dst, self.dst_vocab, padding)

                #trgt = dst[1:] + [self.dst_vocab.word_to_int(self.dst_vocab.end_token)]
                inputs.append(src)
                outputs.append(dst)
                #targets.append(trgt)
                total += 1
            except:
                self.src_file, self.dst_file = self.open_files()

        return np.array(list(map(lambda k: np.array(k), inputs))), np.array(list(map(lambda k: np.array(k), outputs)))
        #, np.array(list(map(lambda k: np.array(k), targets)))

    def _prepare_sentence(self, sentence, vocab, padding):
        tokens = sentence.replace('\n', '').split(' ')
        if padding is not None:
            tokens = tokens + [vocab.end_token] * (padding-len(tokens))
        return list(map(lambda x: vocab.word_to_int(x), tokens))

    def decode_sentence(self, sentence, vocab):
        return list(map(lambda x: vocab.int_to_word(x), sentence))


def remove_repeating_spaces(sentence):
    return re.sub('[ ]{2,}', ' ', sentence)


def remove_non_alphanumeric(sentence):
    return re.sub('[^0-9a-zA-Z ]+', '', sentence)


def remove_diacritics(sentence):
    return unidecode.unidecode(sentence)


def tokenize(sentence):
    return sentence.split(' ')


def prepare_sentence(sentence):
    prepared = remove_repeating_spaces(remove_non_alphanumeric(remove_diacritics(sentence))).strip()
    tokenized = tokenize(sentence)
    return prepared, len(tokenized)


def create_prepared_data(src_sentences_file, dst_sentences_file, prepared_src_file, prepared_dst_file, size=50000,
                         min_length=3, max_length=30):
    with open(src_sentences_file) as source_file:
        with open(dst_sentences_file) as dst_file:
            with open(prepared_src_file, 'w') as output_src_file:
                with open(prepared_dst_file, 'w') as output_dst_file:
                    total = 0
                    while total < size:
                        source_sentence = next(source_file)
                        dst_sentence = next(dst_file)

                        source_sentence, src_len = prepare_sentence(source_sentence)
                        dst_sentence, dst_len = prepare_sentence(dst_sentence)
                        if src_len > max_length or src_len < min_length or dst_len > max_length or dst_len < min_length:
                            continue
                        total += 1
                        output_src_file.write(source_sentence + '\n')
                        output_dst_file.write(dst_sentence + '\n')


if __name__ == '__main__':
    create_prepared_data('data/raw/wiki.pl', 'data/raw/wiki.en', 'data/input_wiki.pl', 'data/input_wiki.en', size=100000)

    src_vocab = Vocabulary.create_from_file('data/input_wiki.pl', vocab_size=10000)
    dst_vocab = Vocabulary.create_from_file('data/input_wiki.en', vocab_size=10000)

    src_vocab.write_to_file('vocab/vocab_wiki.pl')
    dst_vocab.write_to_file('vocab/vocab_wiki.en')

    # data_generator = DataGenerator('vocab/vocab.pl', 'data/input.pl', 'vocab/vocab.en', 'data/input.en')
    # inputs, outputs, targets = data_generator.next_batch(1)
    # print(inputs)
    # print(outputs)
    # print(targets)
    # for input, output, target in zip(input, outputs, targets):
    #     print(input)
    #     print(output)
    #     print(target)
    # print(data_generator.decode_sentence(a, data_generator.src_vocab))
    # print('---')
    # print(b)
    # print(data_generator.decode_sentence(b, data_generator.dst_vocab))
    # print('---')
    # print('---')

