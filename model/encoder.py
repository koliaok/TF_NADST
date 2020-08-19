from utils_s.utils import convert_idx_to_token_tensor
from model.modules import LayerNorm

import tensorflow as tf
import numpy as np

class Encoder(object):

    def __init__(self, fertility_slot_embedding, state_slot_embedding, context_embedding, delex_embedding):
        self.fertility_slot_domain_embedding = fertility_slot_embedding
        self.state_slot_domain_embedding = state_slot_embedding
        self.context_embedding = context_embedding
        self.delex_embedding = delex_embedding


    def get_encoding(self, decoder_type):
        if decoder_type == 'fertility':
            slot_domain_embedding = LayerNorm(self.fertility_slot_domain_embedding, scope="slot_domain_embedding_norm").get_layer_norm()
            context_embedding = LayerNorm(self.context_embedding, scope="context_embedding_norm").get_layer_norm()
            delex_embedding = LayerNorm(self.delex_embedding, scope="delex_embedding_norm").get_layer_norm()
            return slot_domain_embedding, context_embedding, delex_embedding
        else:
            return LayerNorm(self.state_slot_domain_embedding, scope="slot_domain_embedding_norm").get_layer_norm()





class Embedding():
    def __init__(self, d_model, vocab, x, name):
        with tf.compat.v1.variable_scope(name+'_scope', reuse=tf.compat.v1.AUTO_REUSE):
            self.embeddings = tf.compat.v1.get_variable(name,
                                                       dtype=tf.float32,
                                                       shape=(vocab, d_model),
                                                       initializer=tf.initializers.glorot_uniform(),
                                                        )

            self.enc = tf.compat.v1.nn.embedding_lookup(self.embeddings, x)
            self.enc *= d_model ** 0.5

    def get_embedding(self):
        return self.enc

    def get_embedding_wieght(self):
        return self.embeddings


class PositionalEncoding():

    def __init__(self, inputs,
                        maxlen,
                        dropout_rat,
                        masking=True,
                 training=True):

        E = inputs.get_shape().as_list()[-1]
        N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]# dynamic

        # position indices
        # N : Batch size, T: Sequence Length, E: Embedding Size
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) # (N, T)

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i-i%2)/E) for i in range(E)]
            for pos in range(maxlen)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32) # (maxlen, E)

        # lookup
        outputs = tf.compat.v1.nn.embedding_lookup(position_enc, position_ind)

        # masks
        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs) # input 0인 것중에 input 배열 Index에 Input값을 넣고 나머지는 Output으로 대체//////////

        self.outputs = tf.compat.v1.layers.dropout(outputs, rate=dropout_rat, training=training)

    def get_position_encoding(self):
        return self.outputs

