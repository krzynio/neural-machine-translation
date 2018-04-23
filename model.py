#!/usr/bin/env python3
import tensorflow as tf
import logging
import tensorflow.contrib.layers as layers
import numpy as np
from data_utils import DataGenerator, prepare_sentence

BEAM_WIDTH = 5

def seq2seq(mode, features, labels, params):
    src_vocab_size = params['src_vocab_size']
    dst_vocab_size = params['dst_vocab_size']

    embed_dim = params['embed_dim']
    num_units = params['num_units']

    max_length = params['max_length']

    start_token = params['start_token']
    end_token = params['end_token']
    inp = features['input']
    output = features['output']

    batch_size = tf.shape(inp)[0]

    start_tokens = tf.fill([batch_size], start_token)
    train_output = tf.concat([tf.expand_dims(start_tokens, 1), output], 1)
    # input_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(inp, end_token)), 1)
    # output_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(train_output, end_token)), 1)

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

    def beam_decode(scope, beam_width, reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
                encoder_outputs, multiplier=beam_width)
            tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(
                encoder_final_state, multiplier=beam_width)
            tiled_sequence_length = tf.contrib.seq2seq.tile_batch(
                lengths, multiplier=beam_width)

            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=num_units, memory=tiled_encoder_outputs, memory_sequence_length=tiled_sequence_length)
            cell = tf.contrib.rnn.LSTMCell(num_units=num_units)
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell, attention_mechanism, attention_layer_size=num_units)
            out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                attn_cell, dst_vocab_size, reuse=reuse
            )

            decoder_initial_state = attn_cell.zero_state(
                dtype=tf.float32, batch_size=batch_size * beam_width)
            decoder_initial_state = decoder_initial_state.clone(
                cell_state=tiled_encoder_final_state)
            #Tu mozna dodac kare za dlugosc zdania
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=out_cell, embedding=embeddings,
                start_tokens=tf.to_int32(start_tokens), end_token=end_token, initial_state=decoder_initial_state,
                beam_width=beam_width)
            outputs = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder, output_time_major=False,
                impute_finished=True, maximum_iterations=max_length
            )
            return outputs[0]           
            

    train_outputs = decode(train_helper, 'decode')
    pred_outputs = beam_decode('decode', BEAM_WIDTH, True)



    print('-------------------------------------')
    print(pred_outputs)

    tf.identity(train_outputs.sample_id[0], name='train_pred')
    
    loss = None
    train_op = None
    # as mask, we set all valid timesteps as 1 and rest as 0
    print(mode)
    if mode != 'infer':
        weights = tf.to_float(tf.not_equal(train_output[:, :-1], end_token))
        loss = tf.contrib.seq2seq.sequence_loss(
            train_outputs.rnn_output, output, weights=weights)
        train_op = layers.optimize_loss(
            loss, tf.train.get_global_step(),
            optimizer=params.get('optimizer', 'Adam'),
            learning_rate=params.get('learning_rate', 0.001),
            summaries=['loss', 'learning_rate'])

    tf.identity(pred_outputs.sample_id[0], name='predictions')
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_outputs.rnn_output,
        loss=loss,
        train_op=train_op
    )


def make_input_fn(data_generator, batch_size, padding):
    def input_fn():
        inp = tf.placeholder(tf.int32, shape=[None, None], name='input')
        output = tf.placeholder(tf.int32, shape=[None, None], name='output')
        tf.identity(inp[0], 'input_0')
        tf.identity(output[0], 'output_0')
        return {'input': inp, 'output': output}, None

    def feed_fn():
        inputs, outputs = data_generator.next_batch(batch_size, padding)
        #print('feeding')
        return {
            'input:0': inputs,
            'output:0': outputs
        }

    return input_fn, feed_fn


def get_formatter(keys, src_vocab, dst_vocab):
    def to_str(sequence, vocab):
        #try:
        #    sequence = sequence[:sequence.tolist().index(vocab.word_to_int(vocab.end_token))] + [vocab.word_to_int(vocab.end_token)]
        #    return ' '.join([vocab.int_to_word(x) for x in sequence])
        
        tokens = [vocab.int_to_word(x) for x in filter(lambda x: x!= vocab.word_to_int(vocab.start_token) and x != vocab.end_token, sequence)]
        ind = tokens.index(vocab.end_token) if vocab.end_token in tokens else -1
        return ' '.join(tokens[:ind])
        #return ' '.join(tokens)

    def format(values):
        res = []
        res.append('----')
        res.append("[%s]: \t%s" % (keys[0], to_str(values[keys[0]], src_vocab)))
        res.append("[%s]: \t%s" % (keys[1], to_str(values[keys[1]], dst_vocab)))
        result = '\n'.join(res)
        return result

    return format

def predict_test(estimator, data_generator, src_vocab, dst_vocab):
    print(estimator)
    #data = [['czesc, jak masz na imie?']]
    #sentence, _ = prepare_sentence('Ta roznica odpowiada za powod dlaczego on odszedl')
    #data = [sentence]

    def encode(s, padding=45):
        tokens = [src_vocab.start_token] + s.split(' ') + [src_vocab.end_token]
        tokens = tokens + (padding-len(tokens))*[src_vocab.unknown_token]
        return np.array([src_vocab.word_to_int(w) for w in tokens])

    def decode(s, vocab = dst_vocab):
        for token in s:
            w = vocab.int_to_word(token)
            if w == vocab.end_token:
                return
            yield w
    #def input_fn():
    #    inp = tf.Variable(np.array([encode(s) for s in data]), dtype=tf.int32, name='input')
    #    #output = tf.Variable(np.array([encode(s) for s in data]), dtype=tf.int32, name='output')
    #    tf.identity(inp[0], 'input_0')
    #    #tf.identity(output[0], 'output_0')
    #    return {'input': inp, 'output': inp}, None


    #data = np.array([encode(s) for s in data])
    #print(data.shape)
    def input_fn():
        inp, out = data_generator.next_batch(1, 45)
        print('CALLING INPUT_FN')
        for v in range(1):
            print(' '.join(list(decode(inp[v], vocab=src_vocab))))
        #    #print(' '.join(list(decode(out[v], vocab=dst_vocab))))
        #data = 'nie wiem w sumie'
        #encoded = np.array([encode(data)])
        #print(encoded)
        #return {'input': encoded, 'output': encoded}
        return {'input' : inp, 'output' : out}
    total = 0
    #with open('translated.en') as f:

    for _ in range(100):

        for res, _ in zip(estimator.predict(input_fn=input_fn), range(10)):
            results = np.argmax(res, axis=1)
        
            print(' '.join(list(decode(results))))
            return
            #for i in range(10):
            #    print(res[i].shape)
            #    print(res[i])
            #print(' '.join(list(decode(res[i]))))
            #print('----------')

        #print(' '.join(list(decode(res))))
        #print('----')
        #return
            #f.write(' '.join(list(decode(res))))
            #total += 1
            #if total >= 100000:
                
    #print(result)


def predict_loop(estimator, data_generator, padding=45):
    src_vocab = data_generator.src_vocab
    dst_vocab = data_generator.dst_vocab

    def encode(raw_input):
        prepared, _ = prepare_sentence(raw_input)
        return data_generator._prepare_sentence(prepared, src_vocab, padding)

    def decode(s, vocab):
        def decode_gen():
            for token in s:
                w = vocab.int_to_word(token)
                if w == vocab.end_token:
                    return
                yield w
        return ' '.join(decode_gen())

    while 1:
        raw_inp = input('>> ')
        batch = np.array([encode(raw_inp)])
        print('Input: {}'.format(decode(batch[0], src_vocab)))
        result = next(estimator.predict(input_fn=lambda: {'input': batch, 'output': batch}))
        result = np.argmax(result, axis=1)
        print('Prediction: {}'.format(decode(result, dst_vocab)))

if __name__ == '__main__':
    data_generator = DataGenerator('vocab/vocab_subtitles_40000.pl', 'data/input_subtitles_validation.pl', 'vocab/vocab_subtitles_40000.en', 'data/input_subtitles_validation.en')

    params = {
        'src_vocab_size': data_generator.src_vocab.size,
        'dst_vocab_size': data_generator.dst_vocab.size,
        'batch_size': 128,
        'max_length': 45,
        'embed_dim': 300,
        'num_units': 1024,
        'start_token': data_generator.src_vocab.word_to_int(data_generator.src_vocab.start_token),
        'end_token': data_generator.src_vocab.word_to_int(data_generator.src_vocab.end_token)
    }
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.75))
    est = tf.estimator.Estimator(model_fn=seq2seq, model_dir='models/enc-dec-attn-subtitles-40000-lstm', params=params)#, config=config)

    input_fn, feed_fn = make_input_fn(data_generator, params['batch_size'], params['max_length'])

    #tf.logging._logger.setLevel(logging.INFO)

    print_inputs = tf.train.LoggingTensorHook(
        ['input_0', 'output_0'], every_n_iter=100, at_end=False,
        formatter=get_formatter(['input_0', 'output_0'], data_generator.src_vocab, data_generator.dst_vocab))
    print_predictions = tf.train.LoggingTensorHook(
        ['predictions', 'train_pred'], every_n_iter=100,
        formatter=get_formatter(['predictions', 'train_pred'], data_generator.dst_vocab, data_generator.dst_vocab))
    # train
    #est.train(
    #    input_fn=input_fn,
    #    hooks=[tf.train.FeedFnHook(feed_fn), print_inputs, print_predictions])
    # predict
    #est.predict(input_fn=predict_input_fn())
    #predict_test(est, data_generator, data_generator.src_vocab, data_generator.dst_vocab)
    predict_loop(est, data_generator)
