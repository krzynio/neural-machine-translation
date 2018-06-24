#!/usr/bin/env python3
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.layers as layers
from utils import limit
from bleu import compute_bleu
from data import tf_prediction_dataset, tf_train_dataset, tf_multilang_dataset
from utils import load_vocab
from sacremoses import MosesDetokenizer

UNKNOWN_TOKEN = 2
START_TOKEN = 1
END_TOKEN = 0
FR_TOKEN = 3
EN_TOKEN = 4
DE_TOKEN = 5


class TranslatorModel:

    def __init__(self, args, config):
        self.config = config
        self.vocab_encode, self.vocab_decode = load_vocab(args.vocab)
        self.vocab_size = len(self.vocab_encode)
        # self.src_vocab_encode, self.src_vocab_decode = load_vocab(args.src_vocab)
        # self.dst_vocab_encode, self.dst_vocab_decode = load_vocab(args.dst_vocab)
        # self.src_vocab_size = len(self.src_vocab_encode)
        # self.dst_vocab_size = len(self.dst_vocab_encode)
        self.args = args
        self.detokenizer = MosesDetokenizer()
        self.padding = self.args.max_sentence_length + 1
        self.beam_width = args.beam_width
        self.estimator = tf.estimator.Estimator(model_fn=bidirectional_gru_luong,
                                                model_dir=args.model_dir,
                                                params={
                                                    'embed_dim': args.embedding_size,
                                                    'num_units': args.cell_units,
                                                    'max_length': self.padding,
                                                    'vocab_size': self.vocab_size,
                                                    'start_token': START_TOKEN,
                                                    'end_token': END_TOKEN,
                                                    'beam_width': args.beam_width
                                                },
                                                config=config)

    def calculate_bleu(self, src_file, dst_file):
        source = []
        with open(src_file) as f:
            for src in f:
                source.append(src)
        print(len(source))
        references = limit(map(lambda x: [x.split(' ')], open(dst_file)), 20000)
        translations = limit(self.translate(source, return_tokens=True), 20000)
        return compute_bleu(references, translations)

    def translate(self, sentences, return_tokens=False):
        def decode_sentence(tokens):
            if self.beam_width is not None:
                tokens = np.transpose(tokens)[0]
            for t in tokens:
                if t == END_TOKEN:
                    return
                yield self.vocab_decode[t]

        input_fn, init_hook = tf_prediction_dataset(sentences, self.args.src_vocab, 128,
                                                    self.padding, END_TOKEN, UNKNOWN_TOKEN)
        for source, translation in zip(sentences, self.estimator.predict(input_fn=input_fn, hooks=[init_hook])):
            decoded = list(decode_sentence(translation if self.beam_width is not None else np.argmax(translation, axis=1)))
            if return_tokens:
                yield decoded
            else:
                yield (source, self.detokenizer.detokenize(decoded, return_str=True)) 
            #if self.beam_width is not None:
            #    yield source, self.detokenizer.detokenize(decode_sentence(translation), return_str=True)
            #else:
            #    decoded = decode_sentence(np.argmax(translation, axis=1))
            #    yield (source, self.detokenizer.detokenize(decoded, return_str=True) if not return_tokens else decoded)

    def train(self, epochs, log_file='training.log'):

        def load_test_data():
            with open(self.args.src_predict_data) as src_f:
                with open(self.args.dst_predict_data) as dst_f:
                    src_sentences = np.array(src_f.read().split('\n')[:-1])
                    dst_sentences = np.array(dst_f.read().split('\n')[:-1])
                    print(src_sentences)
                    print(dst_sentences)
                    return pd.DataFrame([src_sentences, dst_sentences]).T

        test_data = load_test_data()

        for epoch in range(epochs):
            train_input_fn, train_init_hook = tf_train_dataset(
                self.args.src_train_data,
                self.args.vocab,
                self.args.dst_train_data,
                self.args.vocab,
                batch_size=self.args.batch_size,
                epochs=1,
                padding=self.padding,
                end_token=END_TOKEN,
                unknown_token=UNKNOWN_TOKEN)

            eval_input_fn, eval_init_hook = tf_train_dataset(
                self.args.src_validation_data,
                self.args.vocab,
                self.args.dst_validation_data,
                self.args.vocab,
                batch_size=self.args.batch_size,
                epochs=1,
                padding=self.padding,
                end_token=END_TOKEN,
                unknown_token=UNKNOWN_TOKEN)

            self.estimator.train(input_fn=train_input_fn, hooks=[train_init_hook])

            loss = self.estimator.evaluate(input_fn=eval_input_fn, hooks=[eval_init_hook])

            with open(log_file, 'a') as file:
                file.write('Epoch {}: validation loss = {}\n'.format(epoch, loss))
                to_test = test_data.sample(100)
                src_sentences = to_test[0].as_matrix().flatten()
                dst_sentences = to_test[1].as_matrix().flatten()
                for result, dst in zip(self.translate(src_sentences), dst_sentences):
                    src, translated = result
                    file.write('Input: {}\n'.format(src))
                    file.write('Translation: {}\n'.format(translated))
                    file.write('Target: {}\n'.format(dst))
                    file.write('------\n')
                file.write('\n')


def bidirectional_gru_luong(mode, features, labels, params):
    vocab_size = params['vocab_size']
    # src_vocab_size = params['src_vocab_size']
    # dst_vocab_size = params['dst_vocab_size']
    embed_dim = params['embed_dim']
    num_units = params['num_units']
    max_length = params['max_length']
    start_token = params['start_token']
    end_token = params['end_token']
    beam_width = params['beam_width']

    inp = features['input']
    output = features['output']

    batch_size = tf.shape(inp)[0]

    start_tokens = tf.fill([batch_size], start_token)
    train_output = tf.concat([tf.expand_dims(start_tokens, 1), output], 1)
    lengths = tf.to_int32(tf.fill([batch_size], max_length))

    input_embed = layers.embed_sequence(
        inp, vocab_size=vocab_size, scope='embedding_scope', embed_dim=embed_dim)

    output_embed = layers.embed_sequence(
        train_output, vocab_size=vocab_size, scope='embedding_scope', embed_dim=embed_dim, reuse=True)

    with tf.variable_scope('embedding_scope', reuse=True):
        embeddings = tf.get_variable('embeddings')

    fw_cell = tf.contrib.rnn.LSTMCell(num_units=num_units / 2)
    bw_cell = tf.contrib.rnn.LSTMCell(num_units=num_units / 2)
    encoder_output, encoder_final_state = tf.nn.bidirectional_dynamic_rnn(
        fw_cell,
        bw_cell,
        input_embed,
        dtype=tf.float32
    )
    encoder_output = tf.concat(encoder_output, axis=2)
    if beam_width is not None:
        encoder_final_state = tf.concat(encoder_final_state, axis=1)
    train_helper = tf.contrib.seq2seq.TrainingHelper(output_embed, sequence_length=lengths)
    pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings, start_tokens=tf.to_int32(start_tokens),
                                                           end_token=end_token)

    def decode(helper, scope, reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units=num_units, memory=encoder_output, memory_sequence_length=lengths)
            cell = tf.contrib.rnn.LSTMCell(num_units=num_units)
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell, attention_mechanism, attention_layer_size=num_units / 2)
            out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                attn_cell, vocab_size, reuse=reuse
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

    def beam_decode(scope, reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
                encoder_output, multiplier=beam_width)
            # tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(
            #    encoder_final_state, multiplier=beam_width)
            tiled_sequence_length = tf.contrib.seq2seq.tile_batch(
                lengths, multiplier=beam_width)

            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units=num_units, memory=tiled_encoder_outputs, memory_sequence_length=tiled_sequence_length)
            cell = tf.contrib.rnn.LSTMCell(num_units=num_units)
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell, attention_mechanism, attention_layer_size=num_units / 2)
            out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                attn_cell, vocab_size, reuse=reuse
            )

            decoder_initial_state = attn_cell.zero_state(
                dtype=tf.float32, batch_size=batch_size * beam_width)
            # decoder_initial_state = decoder_initial_state.clone(
            #    cell_state=tiled_encoder_final_state)
            # Tu mozna dodac kare za dlugosc zdania
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=out_cell, embedding=embeddings,
                                                           start_tokens=tf.to_int32(start_tokens),
                                                           end_token=end_token,
                                                           initial_state=decoder_initial_state,
                                                           beam_width=beam_width)
            outputs, state, lens = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder, output_time_major=False,
                impute_finished=False, maximum_iterations=max_length
            )
            return outputs

    train_outputs = decode(train_helper, 'decode')
    if beam_width is not None:
        pred_outputs = beam_decode('decode', reuse=True)
    else:
        pred_outputs = decode(pred_helper, 'decode', reuse=True)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred_outputs.rnn_output if beam_width is None else pred_outputs.predicted_ids
        )

    weights = tf.to_float(tf.not_equal(train_output[:, :-1], end_token))
    loss = tf.contrib.seq2seq.sequence_loss(
        train_outputs.rnn_output, output, weights=weights)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss
        )

    assert mode == tf.estimator.ModeKeys.TRAIN

    train_op = layers.optimize_loss(
        loss, tf.train.get_global_step(),
        optimizer=params.get('optimizer', 'Adam'),
        learning_rate=params.get('learning_rate', 0.001),
        summaries=['loss', 'learning_rate'])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op
    )
