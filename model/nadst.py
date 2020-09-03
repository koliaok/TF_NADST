from model.encoder import Embedding, PositionalEncoding, Encoder
from model.fertility_decoder import Fertility_Decoder
from model.state_decoder import State_Decoder
from model.generator import Generator, PointerGenerator
from model.modules import noam_opt,label_smoothing
from utils_s.config import GATES

import math, copy, time
import tensorflow as tf

class NADST():
    def __init__(self, sess=None):
        """
        For Test Placeholder Setup
        """
        if sess is not None:
            self.sess = sess
            self.contexts = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, None], name="context")
            self.delex_contexts = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, None], name="delex_context")
            self.sorted_in_domainss = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, None],
                                                               name="sorted_in_domain")
            self.sorted_in_slotss = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, None], name="sorted_in_slot")
            self.sorted_in_domains2s = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, None],
                                                                name="sorted_in_domains2")
            self.sorted_in_slots2s = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, None],
                                                              name="sorted_in_slots2")
            self.sorted_generate_ys = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, None],
                                                               name="sorted_generate_y")
            self.sorted_lenvals = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, None], name="sorted_lenval")

    def test_model(self, src_lang, domain_lang, slot_lang, len_val, args, training=True):
        if args['slot_gating']:
            self.sorted_gates = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, None], name="sorted_gate")
        else:
            self.sorted_gates = False

        src_vocab = src_lang.n_words
        slot_vocab = slot_lang.n_words
        domain_vocab = domain_lang.n_words

        with tf.compat.v1.variable_scope('nadst', reuse=tf.compat.v1.AUTO_REUSE):
            fertility_in_domainslots, state_in_domainslots,\
            delex_embedding, contexts_embedding, text_embedding = self.get_embedding(args,
                                                                                     src_vocab,
                                                                                     self.contexts,
                                                                                     self.delex_contexts,
                                                                                     domain_vocab,
                                                                                     self.sorted_in_domainss,
                                                                                     slot_vocab,
                                                                                     self.sorted_in_slotss,
                                                                                     self.sorted_in_domains2s,
                                                                                     self.sorted_in_slots2s,
                                                                                     training=training)

            encoder = Encoder(fertility_slot_embedding=fertility_in_domainslots, state_slot_embedding=state_in_domainslots,
                              delex_embedding=delex_embedding, context_embedding=contexts_embedding)

            self.fertility_generator, self.point_state_generator, gate_gen = self.get_generators(len_val, src_vocab, text_embedding, args)

            self.delex_context_masks_, self.sorted_in_domainslots_masks_, self.context_masks_, src_masks_ = \
                self.get_masks(self.delex_contexts, self.sorted_in_domains2s, self.contexts)


            self.calculation_NADST(encoder=encoder, context_mask=self.context_masks_, delex_context_masks=self.delex_context_masks_,
                                   sorted_in_domainslots_masks=self.sorted_in_domainslots_masks_, args=args,
                                   fertility_generator=self.fertility_generator, state_generator=self.point_state_generator,
                                   src_masks=src_masks_, gate_gen=gate_gen, training=training)

            self.total_loss, self.train_op, self.global_step, self.summaries, self.losses, \
            self.nb_tokens, self.state_out, self.lenval_out_from_gen,\
            self.state_out_from_gen, self.gate_out_from_gen = self.compute_loss(self.sorted_lenvals, self.sorted_gates,
                                                                                  self.contexts, self.context_masks_,
                                                                                  self.sorted_generate_ys,
                                                                                  args['d_model'], 1,
                                                                                  args['warmup'],
                                                                                  gate_gen,
                                                                                  )
            self.evaluation_variable = {'sorted_lenval': self.sorted_lenvals, 'sorted_generate_y': self.sorted_generate_ys,
                                        'sorted_in_domains2': self.sorted_in_domains2s, 'sorted_in_slots2': self.sorted_in_slots2s,
                                        'lenval_out': self.lenval_out_from_gen, 'state_out': self.state_out_from_gen,
                                        'gate_out': self.gate_out_from_gen, 'context': self.contexts, 'sorted_in_domains': self.sorted_in_domainss,
                                        'sorted_in_slots': self.sorted_in_slotss, 'delex_context': self.delex_contexts
                                        }

            if gate_gen is not None:
                self.evaluation_variable['sorted_gates']= self.sorted_gates

        return self.total_loss, self.train_op, self.global_step, self.summaries, self.losses, self.nb_tokens, self.state_out, self.evaluation_variable


    def model(self, xs, ys, src_lang, domain_lang, slot_lang, len_val, args, training=True):
        """
        NADST Model Setup
        """
        # Basic Data
        ids, turn_ids, contexts, delex_context_plains, delex_contexts, context_plains, sorted_in_domainss, turn_beliefs = xs

        if args['slot_gating']:
            sorted_in_domains2s, sorted_in_slotss, sorted_in_slots2s, sorted_lenvals, sorted_generate_ys, context_masks, \
            delex_context_masks, sorted_in_domainslots_masks, sorted_gates = ys
        else:
            sorted_in_domains2s, sorted_in_slotss, sorted_in_slots2s, sorted_lenvals, sorted_generate_ys, context_masks, \
            delex_context_masks, sorted_in_domainslots_masks = ys
            sorted_gates = False

        src_vocab = src_lang.n_words
        slot_vocab = slot_lang.n_words
        domain_vocab = domain_lang.n_words
        with tf.compat.v1.variable_scope('nadst', reuse=tf.compat.v1.AUTO_REUSE):
            # basic embedding setup
            fertility_in_domainslots, state_in_domainslots,\
            delex_embedding, contexts_embedding, text_embedding = self.get_embedding(args,
                                                                                     src_vocab,
                                                                                     contexts,
                                                                                     delex_contexts,
                                                                                     domain_vocab,
                                                                                     sorted_in_domainss,
                                                                                     slot_vocab,
                                                                                     sorted_in_slotss,
                                                                                     sorted_in_domains2s,
                                                                                     sorted_in_slots2s,
                                                                                     training=training)
            # encoder setup
            encoder = Encoder(fertility_slot_embedding=fertility_in_domainslots, state_slot_embedding=state_in_domainslots,
                              delex_embedding=delex_embedding, context_embedding=contexts_embedding)
            # Basic Generator setup (Fertility, Gate, State)
            self.fertility_generator, self.point_state_generator, gate_gen = self.get_generators(len_val, src_vocab, text_embedding, args)
            # Mask data Setup
            delex_context_masks_, sorted_in_domainslots_masks_, context_masks_, src_masks_ = self.get_masks(delex_contexts, sorted_in_domains2s, contexts)

            # Make Fertility, State Decoder
            self.calculation_NADST(encoder=encoder, context_mask=context_masks_, delex_context_masks=delex_context_masks_,
                                   sorted_in_domainslots_masks=sorted_in_domainslots_masks_, args=args,
                                   fertility_generator=self.fertility_generator, state_generator=self.point_state_generator,
                                   src_masks=src_masks_, gate_gen=gate_gen, training=training)
            # Compute Loss Function from 2 Decoder
            total_loss, train_op, global_step, summaries, losses, \
            nb_tokens, state_out, lenval_out_from_gen,\
            state_out_from_gen, gate_out_from_gen = self.compute_loss(sorted_lenvals, sorted_gates,
                                                  contexts, context_masks_,
                                                  sorted_generate_ys,
                                                  args['d_model'], 1,
                                                  args['warmup'],
                                                  gate_gen,
                                                  )

            evaluation_variable = {'sorted_lenval': sorted_lenvals, 'sorted_generate_y': sorted_generate_ys,
                                   'sorted_in_domains2': sorted_in_domains2s, 'sorted_in_slots2': sorted_in_slots2s,
                                   'lenval_out': lenval_out_from_gen, 'state_out': state_out_from_gen,
                                   'gate_out': gate_out_from_gen, 'turn_id': turn_ids,
                                   'context_plain': context_plains, 'ID': ids,
                                   'turn_id': turn_ids, 'turn_belief': turn_beliefs,
                                   'context': contexts, 'context_mask': context_masks,
                                   'sorted_in_domains': sorted_in_domainss, 'sorted_in_slots': sorted_in_slotss,
                                   'sorted_in_domainslots_mask': sorted_in_domainslots_masks, 'delex_context_plain': delex_context_plains,
                                   'delex_context': delex_contexts, 'delex_context_mask': delex_context_masks,
                                   }

            if gate_gen is not None:
                evaluation_variable['sorted_gates']= sorted_gates

        return total_loss, train_op, global_step, summaries, losses, nb_tokens, state_out, evaluation_variable


    def calculation_NADST(self, encoder, context_mask, delex_context_masks, sorted_in_domainslots_masks,
                          args, fertility_generator, state_generator, src_masks, gate_gen=None, training=True):
        """
        Non Autoregressive 2 Decoder(Fertility, State) for basic Transformer(https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)
        """

        self.args = args

        self.fertility_decoder = Fertility_Decoder(encoder, fertility_generator, gate_gen)
        self.fertility_out = self.fertility_decoder.get_fertility_decoder(context_mask, delex_context_masks, args, src_masks, training=training)

        self.state_decoder = State_Decoder(encoder, state_generator)
        self.state_out = self.state_decoder.get_state_decoder(self.fertility_out, context_mask,
                                                             delex_context_masks, sorted_in_domainslots_masks, args, training=training)


    def compute_loss(self, sorted_lenval, sorted_gates, contexts, context_masks, sorted_generate_y, d_model, factor, warmup, gate_gen):
        """
        3 generator(Gate, Fertility, State) loss function
        """
        with tf.compat.v1.variable_scope('comput_loss', reuse=tf.compat.v1.AUTO_REUSE):
            total_loss = 0
            gate_out_loss = tf.compat.v1.constant(0)
            gate_out_nb_tokens = tf.compat.v1.constant(-1)

            # fertitlity decoder out
            slot_out_nb_tokens = tf.shape(tf.compat.v1.reshape(sorted_lenval, shape=[-1]))[-1]
            lenval_out, out_size = self.fertility_decoder.fertility_generator.get_slot_gen(self.state_out['out_slots'])
            one_hot_y = tf.compat.v1.one_hot(sorted_lenval, depth=out_size)
            one_hot_y = tf.compat.v1.reshape(one_hot_y, shape=[-1, out_size])
            lenval_logits = tf.compat.v1.reshape(lenval_out, shape=[-1, out_size])

            fertility_loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=lenval_logits, labels=one_hot_y)
            fertility_loss = tf.compat.v1.reduce_mean(fertility_loss, axis=-1)
            total_loss += fertility_loss

            #gate slot out
            if gate_gen is not None:
                gate_out_nb_tokens = slot_out_nb_tokens
                gate_out, gate_size = self.fertility_decoder.gate_gen.get_slot_gen(self.state_out['out_slots'])
                gate_one_hot_y = tf.compat.v1.one_hot(sorted_gates, depth=gate_size)
                gate_hot_y = tf.compat.v1.reshape(gate_one_hot_y, shape=[-1, gate_size])
                gate_logits = tf.compat.v1.reshape(gate_out, shape=[-1, gate_size])

                gate_out_loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=gate_logits,
                                                                                      labels=gate_hot_y)
                gate_out_loss = tf.compat.v1.reduce_mean(gate_out_loss, axis=-1)
                total_loss += gate_out_loss
            else:
                gate_out = tf.compat.v1.constant(0)
            #state decoder out
            state_out_nb_tokens = tf.shape(tf.compat.v1.reshape(sorted_generate_y, shape=[-1]))[-1]
            self.gen_state_out, state_out_size = self.state_decoder.state_generator.get_point_generator(out_states=self.state_out['out_states'],
                                                                               encoded_context2=self.state_out['encoded_context2'],
                                                                               encoded_in_domainslots2=self.state_out['encoded_in_domainslots2'],
                                                                               contexts=contexts,
                                                                               context_mask=context_masks)
            one_hot_state_y = label_smoothing(tf.compat.v1.reshape(tf.compat.v1.one_hot(sorted_generate_y, depth=state_out_size),
                                                                   shape=[-1, state_out_size]))
            state_logits = tf.compat.v1.reshape(self.gen_state_out, shape=[-1, state_out_size])
            learning_sum = -tf.compat.v1.reduce_sum(one_hot_state_y * state_logits, axis=-1)
            state_loss = tf.compat.v1.reduce_sum(learning_sum)

            #state_loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=state_logits, labels=one_hot_state_y)
            nonpadding = tf.compat.v1.to_float(tf.not_equal(sorted_generate_y, 1))  # 1: <pad>
            nonpadding = tf.compat.v1.reshape(nonpadding, shape=[-1])
            state_loss = tf.reduce_sum(state_loss * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)

            total_loss += state_loss

            losses = {}
            losses['lenval_loss'] = fertility_loss
            losses['state_loss'] = state_loss
            losses['gate_loss'] = gate_out_loss

            nb_tokens = {}
            nb_tokens['slot'] = slot_out_nb_tokens
            nb_tokens['state'] = state_out_nb_tokens
            nb_tokens['gate'] = gate_out_nb_tokens

            global_step = tf.compat.v1.train.get_or_create_global_step()
            lr = noam_opt(d_model=d_model, global_step=global_step, factor=factor, warmup_steps=warmup)
            train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=lr).minimize(total_loss, global_step=global_step)

            tf.compat.v1.summary.scalar('lr', lr)
            tf.compat.v1.summary.scalar("loss", total_loss)
            tf.compat.v1.summary.scalar("global_step", global_step)
            summaries = tf.compat.v1.summary.merge_all()

        return total_loss, train_op, global_step, summaries, losses, nb_tokens, self.state_out, lenval_out, self.gen_state_out, gate_out






    def get_embedding(self, args, src_vocab, contexts, delex_contexts, domain_vocab, sorted_in_domainss, slot_vocab,
                      sorted_in_slotss, sorted_in_domains2s, sorted_in_slots2s, training=True):
        """
        embedding context, delex_context, fertility, state (4)
        """
        # context encoding
        text_embedding = Embedding(args['d_model'], src_vocab, contexts, 'text_embedding')
        contexts_embedding = text_embedding.get_embedding()
        position = PositionalEncoding(contexts_embedding, maxlen=5000, dropout_rat=args['drop'],
                                      training=training).get_position_encoding()
        contexts_embedding += position

        # delex context encoding
        delex_embedding = Embedding(args['d_model'], src_vocab, delex_contexts, 'text_embedding').get_embedding()
        delex_position = PositionalEncoding(delex_embedding, maxlen=5000,
                                            dropout_rat=args['drop'], training=training).get_position_encoding()
        delex_embedding += delex_position

        # fertility domain+slot embedding
        domain_embedding = Embedding(args['d_model'], domain_vocab, sorted_in_domainss,
                                     'domain_embedding').get_embedding()
        domain_position = PositionalEncoding(domain_embedding, maxlen=5000,
                                             dropout_rat=args['drop'], training=training).get_position_encoding()
        domain_embedding += domain_position

        slot_embedding = Embedding(args['d_model'], slot_vocab, sorted_in_slotss, 'slot_embedding').get_embedding()
        slot_position = PositionalEncoding(slot_embedding, maxlen=5000,
                                           dropout_rat=args['drop'], training=training).get_position_encoding()
        slot_embedding += slot_position

        # state domain+slot embedding
        state_domain_embedding = Embedding(args['d_model'], domain_vocab, sorted_in_domains2s,
                                           'domain_embedding').get_embedding()
        state_domain_position = PositionalEncoding(state_domain_embedding, maxlen=5000,
                                                   dropout_rat=args['drop'], training=training).get_position_encoding()
        state_domain_embedding += state_domain_position

        state_slot_embedding = Embedding(args['d_model'], slot_vocab, sorted_in_slots2s,
                                         'slot_embedding').get_embedding()
        state_slot_position = PositionalEncoding(state_slot_embedding, maxlen=5000,
                                                 dropout_rat=args['drop'], training=training).get_position_encoding()
        state_slot_embedding += state_slot_position

        fertility_in_domainslots = domain_embedding + slot_embedding
        state_in_domainslots = state_domain_embedding + state_slot_embedding

        return fertility_in_domainslots, state_in_domainslots,\
        delex_embedding, contexts_embedding, text_embedding

    def get_generators(self, len_val, src_vocab, text_embedding, args):
        """
        Generator Function Gate, Fertility, State
        """
        gate_gen = None
        if args['slot_gating']:
            gate_gen = Generator(len(GATES), scope_name="gate")
        fertility_generator = Generator(len_val + 1, scope_name="fertility")
        state_generator = Generator(src_vocab, text_embedding.get_embedding_wieght(), scope_name='state')
        point_state_generator = PointerGenerator(state_generator=state_generator, scope_name='point_generator')

        return fertility_generator, point_state_generator, gate_gen

    def get_masks(self, delex_contexts, sorted_in_domains2s, contexts):
        """
        make mask Data
        """

        delex_context_masks_ = tf.compat.v1.math.equal(delex_contexts, 1)
        sorted_in_domainslots_masks_ = tf.compat.v1.math.equal(sorted_in_domains2s, 1)
        context_masks_ = tf.compat.v1.math.equal(contexts, 1)
        src_masks_ = tf.compat.v1.zeros_like(sorted_in_domainslots_masks_)  # (N, T1)

        return delex_context_masks_, sorted_in_domainslots_masks_, context_masks_, src_masks_
