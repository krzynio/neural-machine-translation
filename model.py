#!/usr/bin/env python3
import logging

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers


class NMTModel:

    def __init__(self, args, src_encoding, dst_encoding, log_level=logging.INFO):
        tf.logging._logger.setLevel(log_level)
        self.src_encoding = src_encoding
        self.dst_encoding = dst_encoding
        self.args = args
        self.model_location = args.model_location
        self.estimator = tf.estimator.Estimator(model_fn=seq2seq, model_dir=self.model_location, params=self.args)

    def train(self, data_generator, log_every_n_inter=100, steps=None):
        input_fn, feed_fn = self._input_fns(data_generator)
        tensors = ['input_0', 'output_0', 'predictions', 'train_pred']
        logging_hook = tf.train.LoggingTensorHook(
            tensors,
            every_n_iter=log_every_n_inter,
            formatter=NMTModel._create_formatter(tensors, {'input_0': self.src_encoding,
                                                           'output_0': self.dst_encoding,
                                                           'predictions': self.dst_encoding,
                                                           'train_pred': self.dst_encoding})
        )
        self.estimator.train(input_fn=input_fn, hooks=[tf.train.FeedFnHook(feed_fn), logging_hook], steps=steps)

    def predict(self, data):
        batch = self.src_encoding.encode_batch(data, self.args.max_sentence_length)

        def decode_prediction(batch):
            for result, _ in zip(self.estimator.predict(input_fn=lambda: {'input': batch, 'output': batch}),
                                 range(batch.shape[0])):
                yield self.dst_encoding.decode(np.argmax(result, axis=1))

        return list(decode_prediction(batch))

    def _input_fns(self, batch_generator):
        def input_fn():
            inp = tf.placeholder(tf.int32, shape=[None, None], name='input')
            output = tf.placeholder(tf.int32, shape=[None, None], name='output')
            tf.identity(inp[0], 'input_0')
            tf.identity(output[0], 'output_0')
            return {'input': inp, 'output': output}, None

        def feed_fn():
            inputs, outputs = batch_generator.next_batch(self.args.batch_size)
            encoded_inputs = self.src_encoding.encode_batch(inputs, self.args.max_sentence_length)
            encoded_outputs = self.dst_encoding.encode_batch(outputs, self.args.max_sentence_length)
            return {
                'input:0': encoded_inputs,
                'output:0': encoded_outputs
            }

        return input_fn, feed_fn

    @staticmethod
    def _create_formatter(keys, encodings):
        def format(values):
            res = ['']
            for key in keys:
                value = values[key]
                res.append('[{}]: \t {}'.format(key, encodings[key].decode(value)))
            res.append('----')
            return '\n'.join(res)

        return format


def seq2seq(mode, features, labels, params):
    src_vocab_size = params.src_vocab_size
    dst_vocab_size = params.dst_vocab_size
    embed_dim = params.embedding_size
    num_units = params.lstm_units
    max_length = params.max_sentence_length
    start_token = params.start_token
    end_token = params.end_token

    inp = features['input']
    output = features['output']

    batch_size = tf.shape(inp)[0]

    start_tokens = tf.fill([batch_size], start_token)
    train_output = tf.concat([tf.expand_dims(start_tokens, 1), output], 1)

    lengths = tf.to_int32(tf.fill([batch_size], max_length))

    input_embed = layers.embed_sequence(
        inp, vocab_size=src_vocab_size, scope='embed_input', embed_dim=embed_dim)

    output_embed = layers.embed_sequence(
        train_output, vocab_size=dst_vocab_size, scope='embed_output', embed_dim=embed_dim)

    with tf.variable_scope('embed_output', reuse=True):
        embeddings = tf.get_variable('embeddings')

    cell = tf.contrib.rnn.LSTMCell(num_units=num_units)
    encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(cell, input_embed, dtype=tf.float32)

    train_helper = tf.contrib.seq2seq.TrainingHelper(output_embed, sequence_length=lengths)

    pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        embeddings, start_tokens=tf.to_int32(start_tokens), end_token=end_token)

    def decode(helper, scope, reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=num_units, memory=encoder_outputs, memory_sequence_length=lengths)
            cell = tf.contrib.rnn.LSTMCell(num_units=num_units)
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell, attention_mechanism, attention_layer_size=num_units)
            out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                attn_cell, dst_vocab_size, reuse=reuse
            )
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=out_cell, helper=helper,
                initial_state=out_cell.zero_state(
                    dtype=tf.float32, batch_size=batch_size))
            outputs = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder, output_time_major=False,
                impute_finished=True, maximum_iterations=max_length
            )
            return outputs[0]

    train_outputs = decode(train_helper, 'decode')
    pred_outputs = decode(pred_helper, 'decode', reuse=True)

    tf.identity(train_outputs.sample_id[0], name='train_pred')

    loss = None
    train_op = None

    if mode != tf.estimator.ModeKeys.PREDICT:
        weights = tf.to_float(tf.not_equal(train_output[:, :-1], end_token))
        loss = tf.contrib.seq2seq.sequence_loss(
            train_outputs.rnn_output, output, weights=weights)
        train_op = layers.optimize_loss(
            loss, tf.train.get_global_step(),
            optimizer='Adam',
            learning_rate=0.001,
            summaries=['loss', 'learning_rate'])

    tf.identity(pred_outputs.sample_id[0], name='predictions')
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_outputs.rnn_output,
        loss=loss,
        train_op=train_op
    )
