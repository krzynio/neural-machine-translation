#!/usr/bin/env python3

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from dataset import Vocabulary, DataSet, make_input_fn
import logging
from sacremoses import MosesDetokenizer
from model import TranslationModel


class CNNModel(TranslationModel):

    def __init__(self, vocab, args, config):
        self.logger = logging.getLogger('nmt.cnn.model')
        self.vocab = vocab
        self.estimator = estimator = tf.estimator.Estimator(model_fn=cnn,
                                                            model_dir='cnn-test10',
                                                            params={
                                                                'embed_dim': 128,
                                                                'num_units': None,
                                                                'max_length': 50 + 1,
                                                                'vocab_size': 30000,
                                                                'start_token': Vocabulary.START_TOKEN,
                                                                'end_token': Vocabulary.END_TOKEN,
                                                                'beam_width': None
                                                            },
                                                            config=config)
        self.detokenizer = MosesDetokenizer()
        super().__init__(vocab, estimator, 50, self.logger)

    def translate(self, generator, return_tokens=False):
        input_fn, hooks = self.prepare_input(generator)
        for translation in self.estimator.predict(input_fn=input_fn, hooks=hooks):
            token_idx = np.argmax(translation, axis=1)
            tokens = self.vocab.decode_sentence(token_idx, return_tokens=True)
            if return_tokens:
                yield tokens
            else:
                yield self.detokenizer.detokenize(tokens)


def cnn(mode, features, labels, params):
    vocab_size = params['vocab_size']
    embed_dim = params['embed_dim']
    num_units = params['num_units']
    max_length = params['max_length']
    start_token = params['start_token']
    end_token = params['end_token']
    beam_width = params['beam_width']

    inp = features['input']
    output = features['output']

    n_hidden = 512
    n_layers = 4
    dropout_keep_prob = 0.75

    batch_size = tf.shape(inp)[0]

    start_tokens = tf.fill([batch_size], start_token)
    train_output = tf.concat([tf.expand_dims(start_tokens, 1), output], 1)
    lengths = tf.to_int32(tf.fill([batch_size], max_length))

    input_embed = tf.contrib.layers.embed_sequence(
        inp, vocab_size=vocab_size, scope='embedding_layer', embed_dim=embed_dim)

    output_embed = tf.contrib.layers.embed_sequence(
        train_output, vocab_size=vocab_size, scope='embedding_layer', embed_dim=embed_dim, reuse=True)

    with tf.variable_scope('embedding_layer', reuse=True):
        embeddings = tf.get_variable('embeddings')

    def encoder_block(inp, n_hidden, filter_size):
        inp = tf.expand_dims(inp, 2)
        inp = tf.pad(inp, [[0, 0], [(filter_size[0] - 1) // 2, (filter_size[0] - 1) // 2], [0, 0], [0, 0]])
        conv = slim.convolution(inp, n_hidden, filter_size, data_format="NHWC", padding="VALID", activation_fn=None)
        conv = tf.squeeze(conv, 2)
        return conv

    def decoder_block(inp, n_hidden, filter_size):
        inp = tf.expand_dims(inp, 2)
        inp = tf.pad(inp, [[0, 0], [filter_size[0] - 1, 0], [0, 0], [0, 0]])
        conv = slim.convolution(inp, n_hidden, filter_size, data_format="NHWC", padding="VALID", activation_fn=None)
        conv = tf.squeeze(conv, 2)
        return conv

    def glu(x):
        res = tf.multiply(x[:, :, :tf.shape(x)[2] // 2], tf.sigmoid(x[:, :, tf.shape(x)[2] // 2:]))
        return res

    def layer(inp, conv_block, kernel_width, n_hidden, residual=None):
        z = conv_block(inp, n_hidden, (kernel_width, 1))
        res = glu(z) + (residual if residual is not None else 0)
        return res

    def encoder(inp, n_layers):
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            inp = e = slim.linear(tf.nn.dropout(inp, dropout_keep_prob), n_hidden)
            z = None
            for i in range(n_layers):
                z = layer(inp, encoder_block, 3, n_hidden * 2, inp)
                z = tf.nn.dropout(z, dropout_keep_prob)
                inp = z
            return z, z + e

    def decoder(inp, zu, ze, n_layers):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            inp = g = slim.linear(tf.nn.dropout(inp, dropout_keep_prob), n_hidden)
            h = None
            for i in range(n_layers):
                attn_res = h = layer(inp, decoder_block, 3, n_hidden * 2, residual=tf.zeros_like(inp))
                d = slim.linear(h, n_hidden) + g
                dz = tf.matmul(d, tf.transpose(zu, [0, 2, 1]))
                a = tf.nn.softmax(dz)
                c = tf.matmul(a, ze)
                h = slim.linear(attn_res + c, n_hidden)
                h = tf.nn.dropout(h, dropout_keep_prob)
                inp = h
            return h

    zu, ze = encoder(input_embed, n_layers)
    hg = decoder(output_embed, zu, ze, n_layers)

    logits = slim.fully_connected(hg, vocab_size)
    logits = logits[:, :-1]
    pred = tf.nn.softmax(logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred  # pred_outputs.rnn_output if beam_width is None else pred_outputs.predicted_ids
        )

    # weights = tf.to_float(tf.not_equal(train_output[:, :-1], end_token))
    # loss = tf.contrib.seq2seq.sequence_loss(train_outputs.rnn_output, output, weights=weights)
    logits_shape = tf.shape(logits)
    logits = tf.reshape(logits, [logits_shape[0] * logits_shape[1], vocab_size])

    labels = train_output[:, 1:]
    labels = tf.reshape(labels, [-1, ])
    loss_mask = labels > 0
    logits = tf.boolean_mask(logits, loss_mask)
    labels = tf.boolean_mask(labels, loss_mask)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(loss)
    tf.summary.scalar('softmax_loss', loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss
        )

    assert mode == tf.estimator.ModeKeys.TRAIN

    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.train.get_global_step(),
        optimizer=params.get('optimizer', 'Adam'),
        learning_rate=params.get('learning_rate', 0.001),
        summaries=['loss', 'learning_rate'])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op
    )


def main():
    tf.logging._logger.setLevel(logging.INFO)
    config = tf.estimator.RunConfig(
        save_summary_steps=100,
        session_config=None,
        keep_checkpoint_max=1,
        keep_checkpoint_every_n_hours=10000,
        log_step_count_steps=1
    )
    vocab = Vocabulary('data/light_vocab.txt')
    ds = DataSet('data/small_src.txt', 'data/small_src.txt', vocab)

    model = CNNModel(vocab, None, config)
    model.train(ds, validation_dataset=ds, batch_size=8)


if __name__ == '__main__':
    main()
