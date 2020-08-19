import numpy as np

def get_predictions(data,
                    in_domains, in_slots,
                    generated_states,
                    lang, domain_lang, slot_lang, predictions,
                    dontcare_out=[],
                   predicted_delex_context='', gt_delex_context='',
                   generated_lenval=[]):
    domain_lang = domain_lang if domain_lang is not None else lang
    slot_lang = slot_lang if slot_lang is not None else lang
    sorted_index = np.argsort(data['turn_id'])
    for idx in sorted_index:
    #for idx, dial_id in enumerate(data['ID']):
        dial_id = data['ID'][idx].decode('utf-8')
        turn_id = data['turn_id'][idx]
        belief_state = data['turn_belief'][idx]
        state = [lang.index2word[i] for i in generated_states[idx]]
        domains = [domain_lang.index2word[i.item()] for i in in_domains[idx]]
        slots = [slot_lang.index2word[i.item()] for i in in_slots[idx]]
        state_gathered = {}
        for d_idx, domain in enumerate(domains):
            slot = slots[d_idx]
            state_token = state[d_idx]
            if 'PAD' in [domain, slot, state_token]: continue
            if 'EOS' in [domain, slot, state_token]: continue
            if 'SOS' in [domain, slot, state_token]: continue
            if 'UNK' in [domain, slot, state_token]: continue
            if 'dontcare' in [state_token]: continue
            if 'none' in [state_token]: continue
            domain = domain.replace("_DOMAIN", "")
            slot = slot.replace("_SLOT", "")
            key = '{}-{}'.format(domain,slot)
            if key not in state_gathered: state_gathered[key] = ''
            if len(state_gathered[key].split())>0 and state_gathered[key].split()[-1] == state_token: continue
            state_gathered[key] += state_token + ' '
        if len(dontcare_out)>0:
            for out in dontcare_out:
                d, s = out
                domain = domain_lang.index2word[d]
                slot = slot_lang.index2word[s]
                domain = domain.replace("_DOMAIN", "")
                slot = slot.replace("_SLOT", "")
                key = '{}-{}'.format(domain,slot)
                state_gathered[key] = 'dontcare'
        predicted_state = []
        for k,v in state_gathered.items():
            predicted_state.append('{}-{}'.format(k, v.strip()))
        if dial_id not in predictions: predictions[dial_id] = {}
        if turn_id not in predictions[dial_id]: predictions[dial_id][turn_id] = {}
        label_state = []
        for s in belief_state:
            s = s.decode('utf-8')
            v = s.split('-')[-1]
            if v != 'none' and s != 'X':
                label_state.append(s)
        pred_lenval = []
        if len(generated_lenval)>0:
            pred_lenval = ' '.join([str(i) for i in generated_lenval[idx]])

        item = {}
        item['context_plain'] = data['context_plain'][idx].decode('ascii')
        item['delex_context'] = gt_delex_context
        item['predicted_delex_context'] = predicted_delex_context
        item['lenval'] = ' '.join([str(i.item()) for i in data['sorted_lenval'][idx]])
        item['predicted_lenval'] = pred_lenval
        item['turn_belief'] = sorted(label_state)
        item['predicted_belief'] = predicted_state
        predictions[dial_id][turn_id] = item

    return predictions

def predict(state_out, evaluation_variable, lang, domain_lang, slot_lang,
            predictions, oracle, in_lang, args,
            slot_list=None, test_dial_id=None, test_turn_id=-1,
            latency=None, src_lens=None, tgt_lens=None):
    p = args['p_test']  # probability of using the non-ground truth delex context
    ft_p = args['p_test_fertility'] # simulate probability of using the non-ground truth fertility
    if not args['sep_input_embedding']:
        domain_lang = in_lang
        slot_lang = in_lang
    if not oracle:
        pass
    else:
        joint_lenval_acc, joint_gate_acc =0, 0
        #fertility predict
        generated_lenval = np.argmax(evaluation_variable["lenval_out"], axis=-1)
        lenval_compared = (generated_lenval == evaluation_variable['sorted_lenval'])
        res=np.sum((lenval_compared != 1), axis=-1)==0
        joint_lenval_acc = np.sum(np.sum((lenval_compared != 1), axis=-1)==0, axis=-1)

        #gate predict
        if args['slot_gating']:
            generated_gates = np.argmax(evaluation_variable["gate_out"], axis=-1)
            gate_compared = (generated_gates == evaluation_variable['sorted_gates'])
            joint_gate_acc = np.sum(np.sum((gate_compared != 1), axis=-1)==0, axis=-1)

        #state predict
        generated_states = np.argmax(evaluation_variable["state_out"], axis=-1)
        state_compared = (generated_states == evaluation_variable['sorted_generate_y'])

        predictions = get_predictions(data=evaluation_variable, in_domains=evaluation_variable['sorted_in_domains2'],
                                      in_slots=evaluation_variable['sorted_in_slots2'], generated_states=generated_states,
                                      lang=lang, domain_lang=domain_lang, slot_lang=slot_lang, predictions=predictions,
                                      generated_lenval=generated_lenval)

        matches = {}
        matches['joint_lenval'] = joint_lenval_acc
        matches['joint_gate'] = joint_gate_acc
        return matches, predictions
