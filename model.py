#!/usr/bin/env python3
import logging

from bleu import compute_bleu
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from sacremoses import MosesDetokenizer

from dataset import make_input_fn, Vocabulary


class TranslatorModel:
    def __init__(self, args, config):
        self.config = config
        self.vocab = Vocabulary(args.vocab)
        self.args = args
        self.detokenizer = MosesDetokenizer()
        self.max_len = self.args.max_sentence_length
        self.beam_width = args.beam_width
        self.logger = logging.getLogger('nmt.model')
        self.estimator = tf.estimator.Estimator(model_fn=bidirectional_gru_luong,
                                                model_dir=args.model_dir,
                                                params={
                                                    'embed_dim': args.embedding_size,
                                                    'num_units': args.cell_units,
                                                    'max_length': self.max_len + 1,
                                                    'vocab_size': self.vocab.size,
                                                    'start_token': Vocabulary.START_TOKEN,
                                                    'end_token': Vocabulary.END_TOKEN,
                                                    'beam_width': args.beam_width
                                                },
                                                config=config)

    def train(self,
              train_dataset,
              epochs=1,
              batch_size=128,
              validation_dataset=None,
              predict_samples=100):
        self.logger.info('Began training')
        for epoch in range(epochs):
            self.logger.info('Starting epoch {}'.format(epoch + 1))
            input_fn, hooks = self.__prepare_input(train_dataset.new_generator(batch_size=batch_size,
                                                                               max_length=self.max_len))
            self.estimator.train(input_fn=input_fn, hooks=hooks)
            self.logger.info('Epoch {} finished'.format(epoch + 1))

            if validation_dataset is not None:
                self.logger.info('Evaluating validation set')
                validation_loss = self.__evaluate(validation_dataset, batch_size=batch_size)
                self.logger.info('Validation loss: {}'.format(validation_loss))

            if predict_samples is not None:
                generator, src, dst = validation_dataset.new_generator(batch_size=batch_size,
                                                                       max_length=self.max_len,
                                                                       limit=predict_samples,
                                                                       return_raw=True)
                for s, d, t in zip(src, dst, self.translate(generator)):
                    self.logger.info('SRC: {}'.format(s))
                    self.logger.info('DST: {}'.format(d))
                    self.logger.info('TRANSLATION: {}'.format(t))
                    self.logger.info('---')

#            for name, lang in [('DE -> EN', Vocabulary.EN_LANG), ('EN -> FR', Vocabulary.FR_LANG)]:
#                generator, refs = validation_dataset.new_generator_lang_with_refs(batch_size,
#                                                                                  self.max_len,
#                                                                                  lang)
#                for r in refs[:5]:
#                    print(r)
                #refs = list(map(lambda x: x.split(' '), refs))
                #bleu_score = self.calculate_bleu(generator, refs)[0]
                #self.logger.info('{} BLEU: {}'.format(name, bleu_score))
#            return
#        return

    def translate(self, generator, return_tokens=False):
        input_fn, hooks = self.__prepare_input(generator)
        for translation in self.estimator.predict(input_fn=input_fn, hooks=hooks):
            token_idx = np.transpose(translation)[0] if self.beam_width is not None else np.argmax(translation, axis=1)
            tokens = self.vocab.decode_sentence(token_idx, return_tokens=True)
            if return_tokens:
                yield tokens
            else:
                yield self.detokenizer.detokenize(tokens)

    def __evaluate(self, validation_ds, batch_size):
        generator = validation_ds.new_generator(batch_size=batch_size,
                                                max_length=self.max_len)
        input_fn, hooks = self.__prepare_input(generator)
        return self.estimator.evaluate(input_fn=input_fn,
                                       hooks=hooks)

    def __prepare_input(self, generator):
        input_fn, feed_fn = make_input_fn(generator)
        hooks = [tf.train.FeedFnHook(feed_fn)]
        return input_fn, hooks

    def calculate_bleu(self, generator, references):
        translations = self.translate(generator, return_tokens=True)
        references = list(map(lambda x: self.vocab.with_unks(x), references))
        return compute_bleu(references, translations)


def bidirectional_gru_luong(mode, features, labels, params):
    vocab_size = params['vocab_size']
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
        inp, vocab_size=vocab_size, scope='embedding_layer', embed_dim=embed_dim)

    output_embed = layers.embed_sequence(
        train_output, vocab_size=vocab_size, scope='embedding_layer', embed_dim=embed_dim, reuse=True)

    with tf.variable_scope('embedding_layer', reuse=True):
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
