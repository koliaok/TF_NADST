from model.modules import get_layer_connection

import tensorflow as tf

class Fertility_Decoder():

    def __init__(self, encoder, fertility_generator, gate_gen):
        self.encoder = encoder
        self.fertility_generator = fertility_generator
        self.gate_gen = gate_gen

    def get_fertility_decoder(self, context_mask, delex_context_mask, args, src_masks, training=True):
        out = {}
        slot_domain_embedding, context_embedding, delex_embedding = self.encoder.get_encoding(decoder_type='fertility')
        out['encoded_context'] = context_embedding
        out['encoded_delex_context'] = delex_embedding
        out['encoded_in_domainslots'] = slot_domain_embedding

        out_slots = get_layer_connection(slot_domain_embedding, src_masks,
                                   context_embedding, context_mask, delex_embedding, delex_context_mask,
                                   number_head=args['h_attn'],
                                   dropout=args['drop'], line_parameter=args['d_model'], training=training)
        out['out_slots'] = out_slots
        return out
