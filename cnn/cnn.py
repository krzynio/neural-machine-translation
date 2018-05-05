import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.layers as layers

from data import tf_prediction_dataset, tf_train_dataset
from utils import load_vocab
from nltk.tokenize.moses import MosesDetokenizer

UNKNOWN_TOKEN = 2
START_TOKEN = 1
END_TOKEN = 0


class CnnTranslatorModel:

    def __init__(self, args, config):
        self.config = config
        self.src_vocab_encode, self.src_vocab_decode = load_vocab(args.src_vocab)
        self.dst_vocab_encode, self.dst_vocab_decode = load_vocab(args.dst_vocab)
        self.src_vocab_size = len(self.src_vocab_encode)
        self.dst_vocab_size = len(self.dst_vocab_encode)
        self.args = args
        self.detokenizer = MosesDetokenizer()
        self.padding = self.args.max_sentence_length + 1
        self.estimator = tf.estimator.Estimator(model_fn=cnn_seq2seq,
                                                model_dir=args.model_dir,
                                                params={
                                                    'embed_dim': args.embedding_size,
                                                    'pos_embed_count': args.pos_embed_count,
                                                    'cnn_layers': args.cnn_layers,
                                                    'kernel_size': args.kernel_size,
                                                    'conv_size': args.conv_size,
                                                    'max_length': self.padding,
                                                    'src_vocab_size': self.src_vocab_size,
                                                    'dst_vocab_size': self.dst_vocab_size,
                                                    'start_token': START_TOKEN,
                                                    'end_token': END_TOKEN,
                                                    'beam_width' : args.beam_width
                                                },
                                                config=config)

    def translate(self, sentences):
        def decode_sentence(tokens):
            tokens = np.transpose(tokens)
            for t in tokens[0]:
                if t == END_TOKEN:
                    return
                yield self.dst_vocab_decode[t]

        input_fn, init_hook = tf_prediction_dataset(sentences, self.args.src_vocab, 128,
                                                    self.padding, END_TOKEN, UNKNOWN_TOKEN)
        for source, translation in zip(sentences, self.estimator.predict(input_fn=input_fn, hooks=[init_hook])):
            yield source, self.detokenizer.detokenize(decode_sentence(translation), # np.argmax(translation, axis=1)),
                                                      return_str=True)

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
                self.args.src_vocab,
                self.args.dst_train_data,
                self.args.dst_vocab,
                batch_size=self.args.batch_size,
                epochs=1,
                padding=self.padding,
                end_token=END_TOKEN,
                unknown_token=UNKNOWN_TOKEN)

            eval_input_fn, eval_init_hook = tf_train_dataset(
                self.args.src_validation_data,
                self.args.src_vocab,
                self.args.dst_validation_data,
                self.args.dst_vocab,
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

# def source_embeddings(self, inputs_count, embed_dim):
#      return tf.get_variable(name="W", shape=[inputs_count, embed_dim], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))

# def target_embeddings(self, output_count, embed_dim):
#      return tf.get_variable(name="W", shape=[output_count, embed_dim], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))

def cnn_seq2seq(mode, features, labels, params):
    src_vocab_size = params['src_vocab_size']
    dst_vocab_size = params['dst_vocab_size']
    embed_dim = params['embed_dim']
    pos_embed_count = params['pos_embed_count']
    cnn_layers = params['cnn_layers']
    conv_size = params['conv_size']
    kernel_size = params['kernel_size']
    max_length = params['max_length']
    start_token = params['start_token']
    end_token = params['end_token']
    beam_width = params['beam_width']

    inp = features['input']
    output = features['output']

    batch_size = tf.shape(inp)[0]

    input_embed = tf.get_variable(name="inp_emb", shape=[src_vocab_size, embed_dim], initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1))

    input_pos_embed = tf.get_variable(name="inp_pos", shape=[pos_embed_count, embed_dim], initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1))
    output_pos_embed = tf.get_variable(name="inp_pos", shape=[pos_embed_count, embed_dim], initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1))

    output_embed = tf.get_variable(name="out_emb", shape=[dst_vocab_size, embed_dim], initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1))

    # output_embed = layers.embed_sequence(
    #     train_output, vocab_size=dst_vocab_size, scope='embed_output', embed_dim=embed_dim)

    def cnn_reverse(sequence, lengths):
        sequence = tf.reverse_sequence(sequence, lengths, batch_dim=0, seq_dim=1)
        sequence = tf.reverse(tf.reverse(sequence, [1]), [1])
        return sequence

    def get_positional_embedding(lengths, maxlen, embedding_tensor):
        positional_embedding_slice = embedding_tensor[2:maxlen + 2, :]
        batch_size = tf.shape(lengths)[0]
        positional_embedding_batch = tf.tile([positional_embedding_slice], [batch_size, 1, 1])
        #Mask out padded positions
        mask = tf.sequence_mask(lengths=lengths, maxlen=maxlen, dtype=tf.float32)
        positional_embeddings = positional_embedding_batch * tf.expand_dims(mask, 2)
        return cnn_reverse(positional_embeddings, lengths)

    def gated_linear_units(inputs):
        input_shape = inputs.get_shape().as_list()
        assert len(input_shape) == 3
        input_pass = inputs[:,:,0:int(input_shape[2]/2)]
        input_gate = inputs[:,:,int(input_shape[2]/2):]
        input_gate = tf.sigmoid(input_gate)
        return tf.multiply(input_pass, input_gate)

    def conv1d_weightnorm(inputs, layer_idx, out_dim, kernel_size, padding="SAME", dropout=1.0): 
        with tf.variable_scope("conv_layer_"+str(layer_idx)):
            in_dim = int(inputs.get_shape()[-1])
            V = tf.get_variable('V', shape=[kernel_size, in_dim, out_dim], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(4.0*dropout/(kernel_size*in_dim))), trainable=True)
            V_norm = tf.norm(V.initialized_value(), axis=[0,1])  # V shape is M*N*k,  V_norm shape is k  
            g = tf.get_variable('g', dtype=tf.float32, initializer=V_norm, trainable=True)
            b = tf.get_variable('b', shape=[out_dim], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)
            
            # use weight normalization (Salimans & Kingma, 2016)
            W = tf.reshape(g, [1,1,out_dim])*tf.nn.l2_normalize(V,[0,1])
            inputs = tf.nn.bias_add(tf.nn.conv1d(value=inputs, filters=W, stride=1, padding=padding), b)   
            return inputs
            
    def linear_mapping_weightnorm(inputs, out_dim, in_dim=None, dropout=1.0, var_scope_name="linear_mapping"):
        with tf.variable_scope(var_scope_name):
            input_shape = inputs.get_shape().as_list()    # static shape. may has None
            input_shape_tensor = tf.shape(inputs)    
            # use weight normalization (Salimans & Kingma, 2016)  w = g* v/2-norm(v)
            V = tf.get_variable('V', shape=[int(input_shape[-1]), out_dim], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(dropout*1.0/int(input_shape[-1]))), trainable=True)
            V_norm = tf.norm(V.initialized_value(), axis=0)  # V shape is M*N,  V_norm shape is N
            g = tf.get_variable('g', dtype=tf.float32, initializer=V_norm, trainable=True)
            b = tf.get_variable('b', shape=[out_dim], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)   # weightnorm bias is init zero
            
            assert len(input_shape) == 3
            inputs = tf.reshape(inputs, [-1, input_shape[-1]])
            inputs = tf.matmul(inputs, V)
            inputs = tf.reshape(inputs, [input_shape_tensor[0], -1, out_dim])
            #inputs = tf.matmul(inputs, V)    # x*v
            
            scaler = tf.div(g, tf.norm(V, axis=0))   # g/2-norm(v)
            inputs = tf.reshape(scaler,[1, out_dim])*inputs + tf.reshape(b,[1, out_dim])   # x*v g/2-norm(v) + b
    
            return inputs 

    def encode(inputs, lengths):
        inputs = cnn_reverse(inputs, lengths)
        #Get normal embeddings
        embeddings = tf.nn.embedding_lookup(input_embed, inputs)
        #Add positional embeddings
        positional_embeddings = get_positional_embedding(lengths, tf.shape(inputs)[1], input_pos_embed)
        full_embeddings = tf.add(embeddings, positional_embeddings)
        with tf.variable_scope("encoder_cnn"):
            next_layer = full_embeddings
            #Project to size of convolution
            next_layer = linear_mapping_weightnorm(next_layer, conv_size, var_scope_name="mapping_before_convolution")
            for i in range(cnn_layers):
                residual = next_layer
                next_layer = conv1d_weightnorm(next_layer, i, 2 * conv_size, kernel_size)
                next_layer = gated_linear_units(next_layer)
                next_layer = (next_layer + residual) * tf.sqrt(0.5)
            #Project back to size of embedding
            next_layer = linear_mapping_weightnorm(next_layer, embed_dim, var_scope_name="mapping_after_convolution")
            layers.fully_connected(next_layer, embed_dim, activation_fn=None)
        #add output to input embedding for attention
        return next_layer, (next_layer + positional_embeddings) * tf.sqrt(0.5)

    def attention(inputs, target_embeddings, encoder_outputs):
        residual = inputs
        #attention
        inputs = linear_mapping_weightnorm(inputs, embed_dim, var_scope_name="mapping_before_attention")
        inputs = (inputs + target_embeddings) * tf.sqrt(0.5)
        inputs = tf.matmul(inputs, encoder_outputs[0])
        inputs = tf.nn.softmax(inputs)
        attention_scores = inputs

        inputs = tf.matmul(inputs, encoder_outputs[1])
        scale = encoder_outputs[1].shape[1]
        inputs = inputs * (scale * tf.sqrt(1.0 / scale))

        inputs = linear_mapping_weightnorm(inputs, conv_size, var_scope_name="mapping_after_attention")
        inputs = (inputs + residual) * tf.sqrt(0.5)
        return inputs, attention_scores
    
    def split_and_transpose(encoder_outputs):
        encoder_a, encoder_b = encoder_outputs
        encoder_a = tf.transpose(encoder_a, perm=[1,2])
        return encoder_a, encoder_b


    def decode(encoder_outputs, previous_output_tokens, sequence_length, labels):
        encoder_a, encoder_b = split_and_transpose(encoder_outputs)
        next_layer = tf.nn.embedding_lookup(output_embed, previous_output_tokens)
        next_layer += get_positional_embedding(sequence_length, tf.shape(labels)[1], output_pos_embed)
        final_embeddings = next_layer

        # average_attention_scores = None
        with tf.variable_scope("decoder_cnn"):
            #Project to size of convolution
            next_layer = linear_mapping_weightnorm(next_layer, conv_size, var_scope_name="decoder_begin")
            for i in range(cnn_layers):
                residual = next_layer
                next_layer = conv1d_weightnorm(next_layer, i, 2 * conv_size, kernel_size)
                next_layer = gated_linear_units(next_layer)
                #attention
                next_layer, attention_scores = attention(next_layer, final_embeddings, (encoder_a, encoder_b))
                #residual
                next_layer = (next_layer + residual) * tf.sqrt(0.5)
            #Project back to size of vocabulary
            next_layer = linear_mapping_weightnorm(next_layer, embed_dim)
            next_layer = linear_mapping_weightnorm(next_layer, dst_vocab_size)

            return next_layer











    # with tf.variable_scope('embed_output', reuse=True):
    #     embeddings = tf.get_variable('embeddings')


