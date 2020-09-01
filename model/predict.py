import numpy as np
import time, copy

GATES = {"gen":0, "dontcare":1, "none":2}
REVERSE_GATES = {v: k for k, v in GATES.items()}


def make_padding(sequences, pad_token, max_len=-1):
    curr_out = len(sequences)
    for i in range(max_len-curr_out):
        sequences.append(pad_token)
    return [sequences]


def get_input_from_generated(seq1, seq2, seq3, seq4):
    """
    seq1 =  'sorted_in_slots'
    seq2 = generated_lenval
    seq3 = 'sorted_in_domains'
    seq4 = generated_gates
    """

    out_dict = {}
    dontcare_out = []
    for i in range(len(seq1)):
        freq = seq2[i]
        val = seq1[i]
        if seq3 is not None:
            added_val = seq3[i]
        if seq4 is not None:
            gate = seq4[i]
            if gate == GATES['none']:
                continue
            if gate == GATES['dontcare']:
                dontcare_out.append((added_val, val))
                continue
        if val in [0,1,2,3]: continue # slot values as EOS, SOS, UNK, or PAD
        if freq == 0: continue # frequency as zero
        if seq3 is not None:
            if (added_val, val) not in out_dict: out_dict[(added_val, val)] = 0
            out_dict[(added_val, val)] = max([freq, out_dict[(added_val, val)]])
        else:
            if val not in out_dict: out_dict[val] = 0
            out_dict[val] = max([freq, out_dict[val]])
    out = []
    added_out = []
    for val, freq in out_dict.items():
        for j in range(freq):
            if seq3 is not None:
                added_out.append(val[0])
                out.append(val[1])
            else:
                out.append(val)
    if seq3 is not None:
        return added_out, out, dontcare_out
    return out, dontcare_out

def get_delex_from_prediction(turn_data, predictions, in_lang, args):
    turn_id = turn_data['turn_id'][0]
    ID = turn_data['ID'][0]
    gt_delex_context = turn_data['delex_context_plain']
    if ID not in predictions or turn_id-1 not in predictions[ID]:
        return  turn_data['delex_context'], turn_data['delex_context_mask'], gt_delex_context, gt_delex_context
    prev_bs = predictions[ID][turn_id-1]['predicted_belief']
    context = turn_data['context_plain'][0].split()
    delex_context = copy.copy(context)
    gt_delex_context = gt_delex_context.split()
    assert len(context) == len(delex_context) == len(gt_delex_context)
    sys_sos_index = [idx for idx,t in enumerate(delex_context) if t == 'SOS'][1::2]
    user_sos_index = [idx for idx,t in enumerate(delex_context) if t == 'SOS'][::2]

    for bs in prev_bs:
        bs_tokens = bs.split('-')
        d, s, v = bs_tokens[0], bs_tokens[1], '-'.join(bs_tokens[2:])
        ds = '-'.join([d,s])
        if v in ['yes', 'no']:
            if ds == 'hotel-internet':
                v = 'internet wifi'
            elif ds == 'hotel-parking':
                v = 'parking'
            else:
                print(ds, v)
        v_tokens = v.split()
        temp = user_sos_index[:-1]
        for idx, u_idx in enumerate(temp):
            s_idx = sys_sos_index[idx]
            for t_idx, token in enumerate(delex_context[u_idx:s_idx]):
                pos = t_idx + u_idx
                if len(delex_context[pos].split('-')) == 2: continue
                if token in v_tokens:
                    delex_context[pos] = ds
        temp = user_sos_index[1:]
        for idx, u_idx in enumerate(temp):
            s_idx = sys_sos_index[idx]
            for t_idx, token in enumerate(delex_context[s_idx:u_idx]):
                pos = t_idx + s_idx
                delex_context[pos] = gt_delex_context[pos]

    for idx, token in enumerate(delex_context[user_sos_index[-1]:]): # get the original last user uttr
        pos = idx + user_sos_index[-1]
        delex_context[pos] = context[pos]
    out = []
    for token in delex_context:
        token_index = in_lang.word2index[token] if token in in_lang.word2index else in_lang.word2index['UNK']
        out.append(token_index)
    return out, None, ' '.join(delex_context), ' '.join(gt_delex_context)

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

def get_test_predictions(data,
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
        if gt_delex_context:
            idx=0
        dial_id = data['ID'][idx]
        turn_id = data['turn_id'][idx]
        belief_state = data['turn_belief'][idx]
        state = [lang.index2word[i] for i in generated_states[idx]]
        domains = [domain_lang.index2word[i] for i in in_domains[idx]]
        slots = [slot_lang.index2word[i] for i in in_slots[idx]]
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
            v = s.split('-')[-1]
            if v != 'none' and s != 'X':
                label_state.append(s)
        pred_lenval = []
        if len(generated_lenval)>0:
            pred_lenval = ' '.join([str(i) for i in generated_lenval])

        item = {}
        item['context_plain'] = data['context_plain'][idx]
        item['delex_context'] = gt_delex_context
        item['predicted_delex_context'] = predicted_delex_context
        item['lenval'] = ' '.join([str(i) for i in data['sorted_lenval']])
        item['predicted_lenval'] = pred_lenval
        item['turn_belief'] = sorted(label_state)
        item['predicted_belief'] = predicted_state
        predictions[dial_id][turn_id] = item

    return predictions

def predict(state_out, evaluation_variable, lang, domain_lang, slot_lang,
            predictions, oracle, in_lang, args, model=None, feed_dict_func=None,
            slot_list=None, test_dial_id=None, batch_data=None, test_turn_id=-1,
            latency=None, src_lens=None, tgt_lens=None, test=None):

    model_output_list = ["lenval_out", "state_out", "gate_out"]

    p = args['p_test']  # probability of using the non-ground truth delex context
    ft_p = args['p_test_fertility'] # simulate probability of using the non-ground truth fertility
    if not args['sep_input_embedding']:
        domain_lang = in_lang
        slot_lang = in_lang
    if not oracle:
        y_maxlen = model.fertility_generator.out_size
        sorted_index = np.argsort(batch_data['turn_id'])
        c = copy.deepcopy

        for i in sorted_index:
            start = time.time()
            turn_data = {}
            for k,v in batch_data.items():
                turn_data[k] = c(v[i]) if v is not None else v
            if test_dial_id is not None and turn_data['ID'] != test_dial_id: continue
            turn_data['turn_id'] = [turn_data['turn_id']]
            turn_data['ID'] = [turn_data['ID']]
            turn_data['turn_belief'] = [turn_data['turn_belief']]
            turn_data['context'] = [turn_data['context']]
            turn_data['context_mask'] = [turn_data['context_mask']]
            turn_data['sorted_in_domains'] =[turn_data['sorted_in_domains']]
            turn_data['sorted_in_slots'] = [turn_data['sorted_in_slots']]
            turn_data['sorted_lenval'] = [turn_data['sorted_lenval']]
            turn_data['context_plain'] = [turn_data['context_plain']]
            turn_data['sorted_in_domains2'] = [turn_data['sorted_in_domains2']]
            turn_data['sorted_in_slots2'] =[turn_data['sorted_in_slots2']]
            turn_data['sorted_in_domainslots_mask'] = [turn_data['sorted_in_domainslots_mask']]
            turn_data['sorted_generate_y'] = [turn_data['sorted_generate_y']]

            predicted_delex_context = ''
            gt_delex_context = ''

            if model.args['delex_his']:
                if np.random.uniform() < p:
                    delex_context, delex_context_mask, predicted_delex_context, gt_delex_context = get_delex_from_prediction(
                        turn_data, predictions, in_lang, args)
                    turn_data['delex_context'] = [delex_context]
                    turn_data['delex_context_mask'] = [delex_context_mask]
                else:  # use ground truth input
                    turn_data['delex_context'] = [turn_data['delex_context']]
                    turn_data['delex_context_mask'] = [turn_data['delex_context_mask']]
            out = evaluation_variable
            generated_gates = None
            dontcare_out = []
            generated_lenval = []
            if np.random.uniform() < ft_p:
                if args['slot_gating']:
                    generated_gates = np.argmax(out['gate_out'], axis=-1)[i]
                    generated_gates = generated_gates
                if args['slot_lenval']:
                    generated_lenval = np.argmax(out['lenval_out'], axis=-1)[i]

                generated_in_domains2, generated_in_slots2, dontcare_out = get_input_from_generated(
                    turn_data['sorted_in_slots'][0],
                    generated_lenval,
                    turn_data['sorted_in_domains'][0],
                    generated_gates)

                if len(generated_in_domains2) == 0:
                    dial_id = turn_data['ID'][0]
                    turn_id = turn_data['turn_id'][0]
                    if dial_id not in predictions: predictions[dial_id] = {}
                    if turn_id not in predictions[dial_id]: predictions[dial_id][turn_id] = {}
                    label_state = []
                    for s in turn_data['turn_belief'][0]:
                        v = s.split('-')[-1]
                        if v != 'none':
                            label_state.append(s)
                    predictions[dial_id][turn_id]['turn_belief'] = label_state
                    predictions[dial_id][turn_id]['predicted_belief'] = []
                    continue
                generated_in_domains2 = make_padding(generated_in_domains2, pad_token=1,
                                              max_len=len(turn_data['sorted_in_domains2'][0]))
                generated_in_slots2 = make_padding(generated_in_slots2, pad_token=1,
                                              max_len=len(turn_data['sorted_in_slots2'][0]))

                turn_data['sorted_in_domains2'] = generated_in_domains2
                turn_data['sorted_in_slots2'] = generated_in_slots2
                turn_data['sorted_in_domainslots_mask'] = None

            feed_dict = feed_dict_func(model, turn_data)
            gen_state_out = model.sess.run([model.gen_state_out], feed_dict=feed_dict)
            out_attn = None
            if test_dial_id is not None and test_turn_id != -1:
                if turn_data['ID'][0] == test_dial_id and turn_data['turn_id'][0] == test_turn_id:
                    return turn_data

            generated_states = np.argmax(gen_state_out, axis=-1)[0]
            predictions = get_test_predictions(data=turn_data,
                                          in_domains=turn_data['sorted_in_domains2'],
                                          in_slots=turn_data['sorted_in_slots2'],
                                          generated_states=generated_states,
                                          lang=lang, domain_lang=domain_lang, slot_lang=slot_lang,
                                          predictions=predictions, predicted_delex_context=predicted_delex_context,
                                          gt_delex_context=gt_delex_context, generated_lenval=generated_lenval)

            end = time.time()
            elapsed_time = end - start

            src_len = np.sum((np.array(turn_data['context'][0])!= 1), axis=-1)
            tgt_len = np.sum((np.array(generated_in_domains2[0]) != 1), axis=-1)
            if latency is not None and src_lens is not None and tgt_lens is not None:
                latency.append(elapsed_time)
                src_lens.append(src_len)
                tgt_lens.append(tgt_len)
        return predictions, latency, src_lens, tgt_lens

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

        if test is not None:
            predictions = get_test_predictions(data=batch_data,
                                               in_domains=batch_data['sorted_in_domains2'],
                                               in_slots=batch_data['sorted_in_slots2'],
                                               generated_states=generated_states,
                                               lang=lang, domain_lang=domain_lang, slot_lang=slot_lang,
                                               predictions=predictions, generated_lenval=generated_lenval)
        else:
            predictions = get_predictions(data=evaluation_variable,
                                          in_domains=evaluation_variable['sorted_in_domains2'],
                                          in_slots=evaluation_variable['sorted_in_slots2'],
                                          generated_states=generated_states,
                                          lang=lang, domain_lang=domain_lang, slot_lang=slot_lang,
                                          predictions=predictions,
                                          generated_lenval=generated_lenval)

        matches = {}
        matches['joint_lenval'] = joint_lenval_acc
        matches['joint_gate'] = joint_gate_acc
        return matches, predictions
