import numpy as np
import tensorflow as tf
import logging


class Vocabulary:
    END_TOKEN = 0
    START_TOKEN = 1
    UNK_TOKEN = 2

    def __init__(self, path):
        self.__read(path)
        self.size = len(self.__word2ind)

    def encode_word(self, word):
        return self.__word2ind.get(word, Vocabulary.UNK_TOKEN)

    def decode_word(self, ind):
        return self.__ind2word.get(ind)

    def encode_sentence(self, sentence, max_length):
        tokens = sentence.split(' ')
        encoded_tokens = list(map(self.encode_word, tokens))
        padded = encoded_tokens + [Vocabulary.END_TOKEN] * (max_length - len(tokens) + 1)
        return np.array(padded)

    def decode_sentence(self, ids, return_tokens=False):
        def decode(ids):
            for id in ids:
                if id != Vocabulary.END_TOKEN:
                    yield self.decode_word(id)

        tokens = list(decode(ids))
        return tokens if return_tokens else ' '.join(tokens)

    def __read(self, path):
        with open(path) as f:
            words = f.read().split('\n')[:-1]
            self.__word2ind = dict([(v, i) for i, v in enumerate(words)])
            self.__ind2word = dict([(i, v) for i, v in enumerate(words)])


def read_lines(file):
    with open(file) as f:
        return f.read().split('\n')[:-1]


def make_input_fn(generator):
    def input_fn():
        input = tf.placeholder(tf.int32, shape=[None, None], name='input')
        output = tf.placeholder(tf.int32, shape=[None, None], name='output')
        tf.identity(input[0], 'input_0')
        tf.identity(output[0], 'output_0')
        return {'input': input, 'output': output}, None

    def feed_fn():
        src, dst = next(generator)
        return {
            'input:0': src,
            'output:0': dst
        }

    return input_fn, feed_fn


class DataSet:

    def __init__(self, src_file, dst_file, vocab):
        self.vocab = vocab
        self.__read(src_file, dst_file)
        self.logger = logging.getLogger('nmt.dataset_{}_{}'.format(src_file, dst_file))

    def new_generator(self, batch_size, max_length, limit=None, return_raw=False):
        elements = np.random.permutation(list(range(len(self.src_data)))).tolist()
        if limit is not None:
            elements = elements[:limit]
        generator = self.__make_generator(elements, batch_size, max_length)
        if return_raw:
            src_data = [self.src_data[i] for i in elements]
            dst_data = [self.dst_data[i] for i in elements]
            return generator, src_data, dst_data
        else:
            return generator

    def __make_generator(self, elements, batch_size, max_length):
        while len(elements) > 0:
            self.logger.info('Remaining elements: {}'.format(len(elements)))
            batch_idx = elements[:batch_size]
            elements = elements[batch_size:]
            src_batch = [self.src_data[i] for i in batch_idx]
            dst_batch = [self.dst_data[i] for i in batch_idx]
            src = self.encode_batch(src_batch, max_length)
            dst = self.encode_batch(dst_batch, max_length)
            yield (src, dst)

    def encode_batch(self, batch, max_length):
        return np.array(list(map(lambda x: self.vocab.encode_sentence(x, max_length), batch)))

    def __read(self, src_file, dst_file):
        self.src_data = read_lines(src_file)
        self.dst_data = read_lines(dst_file)
        assert len(self.src_data) == len(self.dst_data)

    @staticmethod
    def generator_from_list(sentences, vocab, batch_size, max_length):
        while len(sentences) > 0:
            selected = sentences[:batch_size]
            sentences = sentences[batch_size:]
            encoded = np.array(list(map(lambda x: vocab.encode_sentence(x, max_length), selected)))
            yield (encoded, encoded)
