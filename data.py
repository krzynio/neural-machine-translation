#!/usr/bin/env python3

import tensorflow as tf


class IteratorInitializerHook(tf.train.SessionRunHook):
    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.init_fn = None

    def after_create_session(self, session, coord):
        self.init_fn(session)


def tf_prediction_dataset(sentences, input_vocab, batch_size, padding, end_token, unknown_token):
    init_hook = IteratorInitializerHook()

    def input_fn():
        with open(input_vocab) as f:
            words = f.read().split('\n')[:-1]
            vocab_table = tf.contrib.lookup.HashTable(
                tf.contrib.lookup.KeyValueTensorInitializer(words, list(range(len(words)))), unknown_token)
        input_ds = tf.data.Dataset.from_tensor_slices(sentences) \
            .map(tf_sentence_encoder(vocab_table, padding, end_token))

        ds = tf.data.Dataset.zip((input_ds, input_ds)) \
            .map(tf_to_model_input) \
            .batch(batch_size) \
            .repeat(1)

        it = ds.make_initializable_iterator()

        def init_fn(session):
            session.run(it.initializer)

        init_hook.init_fn = init_fn
        return it.get_next()

    return input_fn, init_hook


def tf_multilang_dataset(input_file, output_file, vocab, batch_size, epochs, padding, end_token, unknown_token):
    init_hook = IteratorInitializerHook()

    def input_fn():
        with open(vocab) as f:
            words = f.read().split('\n')[:-1]
        vocab_table = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(words, list(range(len(words)))), unknown_token)

        input_ds = tf_multilanguage_dataset(input_file, vocab_table, padding, end_token)
        output_ds = tf_multilanguage_dataset(output_file, vocab_table, padding, end_token)

        ds = tf.data.Dataset.zip((input_ds, output_ds)) \
            .map(tf_to_model_input) \
            .batch(batch_size) \
            .repeat(epochs)

        it = ds.make_initializable_iterator()

        def init_fn(session):
            session.run(it.initializer)

        init_hook.init_fn = init_fn
        return it.get_next()

    return input_fn, init_hook


def tf_train_dataset(input_file,
                     input_vocab,
                     output_file,
                     output_vocab,
                     batch_size,
                     epochs,
                     padding,
                     end_token,
                     unknown_token):
    init_hook = IteratorInitializerHook()

    def input_fn():
        input_ds = tf_language_dataset(input_file, input_vocab, padding, end_token, unknown_token)
        output_ds = tf_language_dataset(output_file, output_vocab, padding, end_token, unknown_token)

        ds = tf.data.Dataset.zip((input_ds, output_ds)) \
            .map(tf_to_model_input) \
            .batch(batch_size) \
            .repeat(epochs)

        it = ds.make_initializable_iterator()

        def init_fn(session):
            session.run(it.initializer)

        init_hook.init_fn = init_fn
        return it.get_next()

    return input_fn, init_hook


def tf_to_model_input(input, output):
    return {
        'input': input,
        'output': output
    }


def tf_language_dataset(file_path, vocab_path, padding, end_token, unknown_token):
    with open(vocab_path) as f:
        words = f.read().split('\n')[:-1]
    vocab_table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(words, list(range(len(words)))), unknown_token)
    return tf.data.TextLineDataset(file_path) \
        .map(tf_sentence_encoder(vocab_table, padding, end_token))


def tf_multilanguage_dataset(file_path, vocab_table, padding, end_token):
    return tf.data.TextLineDataset(file_path) \
        .map(tf_sentence_encoder(vocab_table, padding, end_token))


def tf_sentence_encoder(vocab, padding, pad_value):
    def encode_sentence(q):
        q = tf.string_split([q])
        q = tf.sparse_to_dense(q.indices, q.dense_shape, q.values, default_value='')
        q = vocab.lookup(q)
        return tf.pad(
            q,
            [[0, 0], [0, padding - tf.shape(q)[1]]],
            mode='CONSTANT',
            name=None,
            constant_values=pad_value
        )[0]

    return encode_sentence
