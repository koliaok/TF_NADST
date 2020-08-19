from model.modules import LayerNorm, get_layer_connection

import tensorflow as tf

class State_Decoder():

    def __init__(self, encoder, state_generator):
        self.encoder = encoder
        self.state_generator = state_generator

    def get_state_decoder(self, out, context_mask, delex_context_mask, sorted_in_domainslots_masks, args, training=True):
        """
        non auto regressive mode, force trainig mode
        """
        state_slot_domain_embedding2 = self.encoder.get_encoding('state')
        delex_context2 = out['encoded_delex_context']
        context2 = out['encoded_context']

        out_states = get_layer_connection(state_slot_domain_embedding2, sorted_in_domainslots_masks,
                                   context2, context_mask, delex_context2, delex_context_mask, number_head=args['h_attn'],
                                   dropout=args['drop'], line_parameter=args['d_model'], fertility=False, training=training)
        out['out_states'] = out_states
        if args['pointer_decoder']:
            out['encoded_context2'] = context2
            out['encoded_in_domainslots2'] = state_slot_domain_embedding2

        return out