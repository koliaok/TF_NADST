import json

from utils_s.fix_label import *
from utils_s.lang import *
from utils_s.dataset import *
import pdb
import tensorflow as tf
from utils_s.utils import calc_num_batches


def process_turn_belief_dict(turn_belief_dict, turn_domain_flow, ordered_slot):
    domain_numslots_ls = []
    slot_lenval_ls = []
    slotval_ls = []
    in_domain_ls = []
    in_slot_ls = []
    in_domainslot_ls = []
    domain_numslots = {}
    domainslot_lenval = {}
    # (domain, #slot) processing
    for k, v in turn_belief_dict.items():
        d, s = k.split('-')
        if d not in domain_numslots: domain_numslots[d] = 0
        domain_numslots[d] += 1
        if d not in domainslot_lenval: domainslot_lenval[d] = []
        domainslot_lenval[d].append((s, len(v.split())))
    for d in turn_domain_flow:
        if d in domain_numslots:
            domain_numslots_ls.append((d, domain_numslots[d]))
        else:
            domain_numslots_ls.append((d, 0))  # for cases which domain is found but not slot i.e. police domain
    # (slot, len_slotVal) processing 
    if ordered_slot == 'alphabetical':
        for k, v in domainslot_lenval.items():
            sorted_v = sorted(v, key=lambda tup: tup[0])
            domainslot_lenval[k] = sorted_v
    for dn in domain_numslots_ls:
        domain, numslots = dn
        for n in range(numslots):
            slot_lenval_ls.append((domainslot_lenval[domain][n]))
            in_domain_ls.append(domain)
    # (domain_slot, Val) processing 
    for i, d in enumerate(in_domain_ls):
        s, len_v = slot_lenval_ls[i]
        slotval_ls += turn_belief_dict["{}-{}".format(d, s)].split()
        for l in range(len_v):
            in_domainslot_ls.append((d, s))
    assert len(in_domain_ls) == len(slot_lenval_ls)
    assert len(in_domainslot_ls) == len(slotval_ls)
    return domain_numslots_ls, in_domain_ls, slot_lenval_ls, in_domainslot_ls, slotval_ls


def fix_book_slot_name(turn_belief_dict, slots):
    out = {}
    for k in turn_belief_dict.keys():
        new_k = k.replace(" ", "")
        if new_k not in slots: pdb.set_trace()
        out[new_k] = turn_belief_dict[k]
    return out


def fix_multival(turn_belief_dict, multival_count):
    has_multival = False
    for k, v in turn_belief_dict.items():
        if '|' in v:
            values = v.split('|')
            turn_belief_dict[k] = values[0]  # ' ; '.join(values)
            has_multival = True
    if has_multival: multival_count += 1
    return turn_belief_dict, multival_count


def remove_none_value(turn_belief_dict):
    out = {}
    for k, v in turn_belief_dict.items():
        if v != 'none':
            out[k] = v
    return out


def get_sorted_lenval(sorted_domainslots, turn_belief_dict, slot_gating):
    sorted_lenval = [0] * len(sorted_domainslots)
    if slot_gating:
        sorted_gates = [GATES['none']] * len(sorted_domainslots)
    else:
        sorted_gates = None
    for k, v in turn_belief_dict.items():
        index = sorted_domainslots.index(k)
        lenval = len(v.split())
        if lenval >= 9:
            print('as')
        if not slot_gating or (slot_gating and v not in ['dontcare', 'none']):
            sorted_lenval[index] = lenval
        if slot_gating:
            if v not in ['dontcare', 'none']:
                sorted_gates[index] = GATES['gen']
            else:
                sorted_gates[index] = GATES[v]
    return sorted_lenval, sorted_gates


def get_sorted_generate_y(sorted_domainslots, sorted_lenval_ls, turn_belief_dict):
    in_domains = []
    in_slots = []
    in_domainslots_index = []
    out_vals = []
    for idx, lenval in enumerate(sorted_lenval_ls):
        if lenval == 0: continue
        domain = sorted_domainslots[idx].split('-')[0]
        slot = sorted_domainslots[idx].split('-')[1]
        val = turn_belief_dict[sorted_domainslots[idx]].split()
        assert len(val) == lenval
        for i in range(lenval):
            in_domains.append(domain + "_DOMAIN")
            in_slots.append(slot + "_SLOT")
            out_vals.append(val[i])
            in_domainslots_index.append(idx)
    return in_domains, in_slots, out_vals, in_domainslots_index


'''
def get_bs_seq(sorted_domainslots, turn_belief_dict):
    bs_seq = []
    for ds in sorted_domainslots:
        if ds in turn_belief_dict:
            d,s = ds.split('-')
            v = turn_belief_dict[ds]
            bs_seq.append('{}_DOMAIN'.format(d))
            bs_seq.append('{}_SLOT'.format(s))
            bs_seq += v.split()
    bs_seq = ' '.join(bs_seq)
    return bs_seq
'''


def get_atrg_generate_y(sorted_domainslots, sorted_lenval_ls, turn_belief_dict):
    vals = []
    indices = []
    for idx, lenval in enumerate(sorted_lenval_ls):
        if lenval == 0: continue
        val = turn_belief_dict[sorted_domainslots[idx]].split()
        assert len(val) == lenval
        vals.append(val)
        indices.append(idx)
    return vals, indices


def read_langs(file_name, SLOTS, dataset, lang, mem_lang, training, args):
    print(("Reading from {}".format(file_name)))
    data = []
    max_len_val_per_slot = 0
    max_len_slot_val = {}
    domain_counter = {}
    # count_noise = 0
    sorted_domainslots = sorted(SLOTS)
    sorted_in_domains = [i.split('-')[0] + "_DOMAIN" for i in sorted_domainslots]
    sorted_in_slots = [i.split('-')[1] + "_SLOT" for i in sorted_domainslots]
    for ds in sorted_domainslots:
        max_len_slot_val[ds] = (1, "none")  # counting none/dontcare
    multival_count = 0

    with open(file_name) as f:
        dials = json.load(f)
        # create vocab first
        for dial_dict in dials:
            if (dataset == 'train' and training) or (args['pointer_decoder']):
                for ti, turn in enumerate(dial_dict["dialogue"]):
                    lang.index_words(turn["system_transcript"], 'utter')
                    lang.index_words(turn["transcript"], 'utter')

        for dial_dict in dials:
            last_belief_dict = {}
            # Filtering and counting domains
            for domain in dial_dict["domains"]:
                if domain not in domain_counter.keys():
                    domain_counter[domain] = 0
                domain_counter[domain] += 1

            # Reading data
            dialog_history = ''
            delex_dialog_history = ''
            prev_turn_belief_dict = {}

            for ti, turn in enumerate(dial_dict["dialogue"]):
                turn_id = turn["turn_idx"]
                if ti == 0:
                    user_sent = ' SOS ' + turn["transcript"] + ' EOS '
                    sys_sent = ''
                    dlx_user_sent = ' SOS ' + turn["delex_transcript"] + ' EOS '
                    dlx_sys_sent = ''
                else:
                    sys_sent = ' SOS ' + turn["system_transcript"] + ' EOS '
                    user_sent = 'SOS ' + turn["transcript"] + ' EOS '
                    dlx_sys_sent = ' SOS ' + turn["delex_system_transcript"] + ' EOS '
                    dlx_user_sent = 'SOS ' + turn["delex_transcript"] + ' EOS '
                turn_uttr = sys_sent + user_sent
                dialog_history += sys_sent
                dialog_history += user_sent
                delex_dialog_history += dlx_sys_sent
                delex_dialog_history += dlx_user_sent

                turn_belief_dict = fix_general_label_error(turn["belief_state"], False, SLOTS)
                turn_belief_dict = fix_book_slot_name(turn_belief_dict, SLOTS)
                turn_belief_dict, multival_count = fix_multival(turn_belief_dict, multival_count)
                turn_belief_dict = remove_none_value(turn_belief_dict)

                sorted_lenval, sorted_gates = get_sorted_lenval(sorted_domainslots, turn_belief_dict,
                                                                args['slot_gating'])
                sorted_in_domains2, sorted_in_slots2, sorted_generate_y, sorted_in_domainslots2_index = get_sorted_generate_y(
                    sorted_domainslots, sorted_lenval, turn_belief_dict)

                if args['auto_regressive']:
                    atrg_generate_y, sorted_in_domainslots2_index = get_atrg_generate_y(sorted_domainslots,
                                                                                        sorted_lenval, turn_belief_dict)
                else:
                    atrg_generate_y = None

                if args['delex_his']:
                    temp = dialog_history.split()
                    delex_temp = delex_dialog_history.split()
                    start_idx = [i for i, t in enumerate(temp) if t == 'SOS'][
                        -1]  # delex all except the last user utterance
                    in_delex_dialog_history = ' '.join(delex_temp[:start_idx] + temp[start_idx:])
                    if len(in_delex_dialog_history.split()) != len(dialog_history.split()): pdb.set_trace()
                    if (dataset == 'train' and training) or (args['pointer_decoder']):
                        lang.index_words(in_delex_dialog_history, 'utter')

                turn_belief_list = [str(k) + '-' + str(v) for k, v in turn_belief_dict.items()]
                for k, v in turn_belief_dict.items():
                    if len(v.split()) > max_len_slot_val[k][0]:
                        max_len_slot_val[k] = (len(v.split()), v)

                if dataset == 'train' and training:
                    mem_lang.index_words(turn_belief_dict, 'belief')

                data_detail = {
                    "ID": dial_dict["dialogue_idx"],
                    "turn_id": turn_id,
                    "dialog_history": dialog_history.strip(),
                    "delex_dialog_history": in_delex_dialog_history.strip(),
                    "turn_belief": turn_belief_list,
                    "sorted_domainslots": sorted_domainslots,
                    "turn_belief_dict": turn_belief_dict,
                    "turn_uttr": turn_uttr.strip(),
                    'sorted_in_domains': sorted_in_domains,
                    'sorted_in_slots': sorted_in_slots,
                    'sorted_in_domains2': sorted_in_domains2,
                    'sorted_in_slots2': sorted_in_slots2,
                    'sorted_in_domainslots2_idx': sorted_in_domainslots2_index,
                    'sorted_lenval': sorted_lenval,
                    'sorted_gates': sorted_gates,
                    'sorted_generate_y': sorted_generate_y,
                    'atrg_generate_y': atrg_generate_y
                }
                data.append(data_detail)
                if len(sorted_lenval) > 0 and max(sorted_lenval) > max_len_val_per_slot:
                    max_len_val_per_slot = max(sorted_lenval)
                prev_turn_belief_dict = turn_belief_dict

    print("domain_counter", domain_counter)
    print("multival_count", multival_count)

    return data, SLOTS, max_len_val_per_slot, max_len_slot_val




def data_reprocessing(dataset, batch_size):
    data = [dataset[i] for i in range(len(dataset))]
    last_out = collate_fn(data, batch_size)
    #out_data = transfer_data_info_type(last_out)
    return last_out


def generator_fn(ID, turn_id, context, delex_context_plain, delex_context, context_plain, sorted_in_domains,
                 turn_belief, sorted_in_domains2, sorted_in_slots, sorted_in_slots2, sorted_lenval, sorted_generate_y, context_mask,
                 delex_context_mask, sorted_in_domainslots_mask):
    for ids, turn_ids, contexts, delex_context_plains, delex_contexts, context_plains, sorted_in_domainss, turn_beliefs, \
        sorted_in_domains2s, sorted_in_slotss, sorted_in_slots2s, sorted_lenvals, sorted_generate_ys, \
        context_masks, delex_context_masks, sorted_in_domainslots_masks in zip(ID, turn_id, context,
                                                                               delex_context_plain,
                                                                               delex_context, context_plain,
                                                                               sorted_in_domains, turn_belief,
                                                                               sorted_in_domains2, sorted_in_slots,
                                                                               sorted_in_slots2,
                                                                               sorted_lenval, sorted_generate_y,
                                                                               context_mask,
                                                                               delex_context_mask,
                                                                               sorted_in_domainslots_mask):
        yield (ids, turn_ids, contexts, delex_context_plains,
               delex_contexts, context_plains, sorted_in_domainss, turn_beliefs), \
              (sorted_in_domains2s, sorted_in_slotss, sorted_in_slots2s,
               sorted_lenvals, sorted_generate_ys, context_masks,
               delex_context_masks, sorted_in_domainslots_masks)

def generator_fn_slot_gate(ID, turn_id, context, delex_context_plain, delex_context, context_plain, sorted_in_domains,
                 turn_belief, sorted_in_domains2, sorted_in_slots, sorted_in_slots2, sorted_lenval, sorted_generate_y, context_mask,
                 delex_context_mask, sorted_in_domainslots_mask, sorted_gates):
    for ids, turn_ids, contexts, delex_context_plains, delex_contexts, context_plains, sorted_in_domainss, turn_beliefs, \
        sorted_in_domains2s, sorted_in_slotss, sorted_in_slots2s, sorted_lenvals, sorted_generate_ys, \
        context_masks, delex_context_masks, sorted_in_domainslots_masks, sorted_gatess in zip(ID, turn_id, context,
                                                                                               delex_context_plain,
                                                                                               delex_context, context_plain,
                                                                                               sorted_in_domains, turn_belief,
                                                                                               sorted_in_domains2, sorted_in_slots,
                                                                                               sorted_in_slots2,
                                                                                               sorted_lenval, sorted_generate_y,
                                                                                               context_mask,
                                                                                               delex_context_mask,
                                                                                               sorted_in_domainslots_mask, sorted_gates):
        yield (ids, turn_ids, contexts, delex_context_plains,
               delex_contexts, context_plains, sorted_in_domainss, turn_beliefs), \
              (sorted_in_domains2s, sorted_in_slotss, sorted_in_slots2s,
               sorted_lenvals, sorted_generate_ys, context_masks,
               delex_context_masks, sorted_in_domainslots_masks, sorted_gatess)


def get_eager_tensor_dataset(total_data_info, batch_size, slot_gating):
    end = batch_size
    ID = tf.compat.v1.constant(dtype=tf.string, value=total_data_info['ID'][:end])
    turn_id = tf.compat.v1.constant(dtype=tf.int32, value=total_data_info['turn_id'][:end])
    context = tf.compat.v1.constant(dtype=tf.int32, value=total_data_info['context'][:end])
    delex_context_plain = tf.compat.v1.constant(dtype=tf.string, value=total_data_info['delex_context_plain'][:end])
    delex_context = tf.compat.v1.constant(dtype=tf.int32, value=total_data_info['delex_context'][:end])
    context_plain = tf.compat.v1.constant(dtype=tf.string, value=total_data_info['context_plain'][:end])
    sorted_in_domains = tf.compat.v1.constant(dtype=tf.int32, value=total_data_info['sorted_in_domains'][:end])
    turn_belief = tf.compat.v1.constant(dtype=tf.string, value=total_data_info['turn_belief'][:end])
    sorted_in_domains2 = tf.compat.v1.constant(dtype=tf.int32, value=total_data_info['sorted_in_domains2'][:end])
    sorted_in_slots = tf.compat.v1.constant(dtype=tf.int32, value=total_data_info['sorted_in_slots'][:end])
    sorted_in_slots2 = tf.compat.v1.constant(dtype=tf.int32, value=total_data_info['sorted_in_slots2'][:end])
    sorted_lenval = tf.compat.v1.constant(dtype=tf.int32, value=total_data_info['sorted_lenval'][:end])
    sorted_generate_y = tf.compat.v1.constant(dtype=tf.int32, value=total_data_info['sorted_generate_y'][:end])
    context_mask = tf.compat.v1.constant(dtype=tf.bool, value=total_data_info['context_mask'][:end])
    delex_context_mask = tf.compat.v1.constant(dtype=tf.bool, value=total_data_info['delex_context_mask'][:end])
    sorted_in_domainslots_mask = tf.compat.v1.constant(dtype=tf.bool, value=total_data_info['sorted_in_domainslots_mask'][:end])

    if slot_gating:
        sorted_gates = tf.compat.v1.constant(dtype=tf.int32, value=total_data_info['sorted_gates'][:end])

        xs = (ID, turn_id, context, delex_context_plain, delex_context, context_plain, sorted_in_domains, turn_belief)
        ys = (sorted_in_domains2, sorted_in_slots, sorted_in_slots2, sorted_lenval, sorted_generate_y, context_mask,
              delex_context_mask, sorted_in_domainslots_mask, sorted_gates)
    else:
        xs = (ID, turn_id, context, delex_context_plain, delex_context, context_plain, sorted_in_domains, turn_belief)
        ys = (sorted_in_domains2, sorted_in_slots, sorted_in_slots2, sorted_lenval, sorted_generate_y, context_mask,
              delex_context_mask, sorted_in_domainslots_mask)

    return xs, ys


def get_tensor_dataset(total_data_info, batch_size, slot_gating, shuffle):
    end = 100#len(total_data_info['ID'])
    ID = total_data_info['ID'][:end]
    turn_id = total_data_info['turn_id'][:end]
    context = total_data_info['context'][:end]
    delex_context_plain = total_data_info['delex_context_plain'][:end]
    delex_context = total_data_info['delex_context'][:end]
    context_plain = total_data_info['context_plain'][:end]
    sorted_in_domains = total_data_info['sorted_in_domains'][:end]
    turn_belief = total_data_info['turn_belief'][:end]
    sorted_in_domains2 = total_data_info['sorted_in_domains2'][:end]
    sorted_in_slots = total_data_info['sorted_in_slots'][:end]
    sorted_in_slots2 = total_data_info['sorted_in_slots2'][:end]
    sorted_lenval = total_data_info['sorted_lenval'][:end]
    sorted_generate_y = total_data_info['sorted_generate_y'][:end]
    context_mask = total_data_info['context_mask'][:end]
    delex_context_mask = total_data_info['delex_context_mask'][:end]
    sorted_in_domainslots_mask = total_data_info['sorted_in_domainslots_mask'][:end]
    if slot_gating:
        sorted_gates = total_data_info["sorted_gates"][:end]

        """
           Returns
        1 line : ID, turn_id, turn_belief, context, delex_context_plain, delex_context, context_plain, sorted_in_domains, turn_belief -> 8개 
        2 line : sorted_in_domains2, sorted_in_slots, sorted_in_slots2, sorted_lenval, sorted_generate_y, context_mask, delex_context_mask, sorted_in_domainslots_mask, sorted_gates -> 9개
        """
        shapes = (((), (), [None], (), [None], (), [None], [None]),  # -> 8
                  ([None], [None], [None], [None], [None], [None], [None], [None], [None]))  # -> 9
        types = ((tf.string, tf.int32, tf.int32, tf.string, tf.int32, tf.string, tf.int32, tf.string),
                 (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.bool, tf.bool, tf.bool, tf.int32))
        dataset = tf.data.Dataset.from_generator(
            generator_fn_slot_gate,
            output_shapes=shapes,
            output_types=types,
            args=(ID, turn_id, context, delex_context_plain, delex_context, context_plain, sorted_in_domains, turn_belief,
                  sorted_in_domains2, sorted_in_slots, sorted_in_slots2, sorted_lenval, sorted_generate_y, context_mask,
                  delex_context_mask, sorted_in_domainslots_mask, sorted_gates))  # <- arguments for generator_fn. converted to np string arrays
    else:
        """
           Returns
        1 line : ID, turn_id, turn_belief, context, delex_context_plain, delex_context, context_plain, sorted_in_domains, turn_belief -> 8개 
        2 line : sorted_in_domains2, sorted_in_slots, sorted_in_slots2, sorted_lenval, sorted_generate_y, context_mask, delex_context_mask, sorted_in_domainslots_mask, sorted_gates -> 9개
        """
        shapes = (((), (), [None], (), [None], (), [None], [None]),# -> 8
                  ([None], [None], [None], [None], [None], [None], [None], [None]))# -> 8
        types = ((tf.string, tf.int32, tf.int32, tf.string, tf.int32, tf.string, tf.int32, tf.string),
                 (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.bool, tf.bool, tf.bool))
        dataset = tf.data.Dataset.from_generator(
            generator_fn,
            output_shapes=shapes,
            output_types=types,
            args=(ID, turn_id, context, delex_context_plain, delex_context, context_plain, sorted_in_domains, turn_belief,
                  sorted_in_domains2, sorted_in_slots, sorted_in_slots2, sorted_lenval, sorted_generate_y, context_mask,
                  delex_context_mask, sorted_in_domainslots_mask))  # <- arguments for generator_fn. converted to np string arrays


    if shuffle:  # for training
        dataset = dataset.shuffle(128 * batch_size)

    dataset = dataset.repeat()  # iterate forever
    dataset = dataset.batch(batch_size).prefetch(32)
    num_batches = calc_num_batches(len(context), batch_size)
    return dataset, num_batches, len(context)


def get_seq(pairs, lang, mem_lang, domain_lang, slot_lang, args, split, batch_size, ALL_SLOTS):
    data_info = {}
    data_keys = pairs[0].keys()
    for k in data_keys:
        data_info[k] = []

    for pair in pairs:
        for k in data_keys:
            data_info[k].append(pair[k])

    dataset = Dataset(data_info, lang, mem_lang, domain_lang, slot_lang, args, split, ALL_SLOTS)
    total_data_info = data_reprocessing(dataset, batch_size)
    return total_data_info


def get_slot_information(ontology):
    # if not EXPERIMENT_DOMAINS:
    ontology_domains = dict([(k, v) for k, v in ontology.items()])
    # else:
    # ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    slots = [k.replace(" ", "").lower() for k in ontology_domains.keys()]
    all_domains = [i.split('-')[0] for i in slots]
    all_domains = set(all_domains)
    return slots, all_domains


def merge_lang(lang, max_freq):
    out = Lang()
    for k, v in lang.index2word.items():
        if k < 4:
            continue
        else:
            for f in range(max_freq + 1):  # including length/size 0
                out.index_word((v, f))
    return out


def prepare_data_seq(training, args):
    batch_size = args['batch']
    file_train = 'data{}/nadst_train_dials.json'.format(args['data_version'])
    file_dev = 'data{}/nadst_dev_dials.json'.format(args['data_version'])
    file_test = 'data{}/nadst_test_dials.json'.format(args['data_version'])
    ontology = json.load(open("data2.0/multi-woz/MULTIWOZ2 2/ontology.json", 'r'))
    ALL_SLOTS, ALL_DOMAINS = get_slot_information(ontology)
    lang, mem_lang = Lang(), Lang()
    domain_lang, slot_lang = Lang(), Lang()
    lang.index_words(ALL_SLOTS, 'slot')
    mem_lang.index_words(ALL_SLOTS, 'slot')
    lang.index_word('dontcare')
    mem_lang.index_word('dontcare')
    domain_lang.index_words(ALL_SLOTS, 'domain_only')
    slot_lang.index_words(ALL_SLOTS, 'slot_only')

    if training:
        pair_train, slot_train, train_max_len_val, train_max_len_slot_val = read_langs(file_train, ALL_SLOTS, "train",
                                                                                       lang, mem_lang, training, args)
        pair_dev, slot_dev, dev_max_len_val, dev_max_len_slot_val = read_langs(file_dev, ALL_SLOTS, "dev", lang,
                                                                               mem_lang, training, args)
        pair_test, slot_test, test_max_len_val, test_max_len_slot_val = read_langs(file_test, ALL_SLOTS, "test", lang,
                                                                                   mem_lang, training, args)
        max_len_val = max(train_max_len_val, dev_max_len_val, test_max_len_val)

        if not args['sep_input_embedding']:
            lang.index_words(domain_lang.word2index, 'domain_w2i')
            lang.index_words(slot_lang.word2index, 'slot_w2i')

        train_total_info = get_seq(pair_train, lang, mem_lang, domain_lang, slot_lang, args, 'train', args['batch'], ALL_SLOTS)
        dev_total_info = get_seq(pair_dev, lang, mem_lang, domain_lang, slot_lang, args, 'dev', args['eval_batch'], ALL_SLOTS)
        test_total_info = get_seq(pair_test, lang, mem_lang, domain_lang, slot_lang, args, 'test', args['eval_batch'], ALL_SLOTS)

    data_info_dict = {"train": train_total_info,
                      "dev": dev_total_info,
                      "test": test_total_info}

    print("Read %s pairs train" % len(pair_train))
    print("Read %s pairs dev" % len(pair_dev))
    print("Read %s pairs test" % len(pair_test))

    print("Vocab_size: %s " % lang.n_words)
    print("Vocab_size Training %s" % lang.n_words)
    print("Vocab_size Belief %s" % mem_lang.n_words)
    print("Vocab_size Domain {}".format(domain_lang.n_words))
    print("Vocab_size Slot {}".format(slot_lang.n_words))

    print("Max. len of value per slot: train {} dev {} test {} all {}".format(train_max_len_val, dev_max_len_val,
                                                                              test_max_len_val, max_len_val))

    SLOTS_LIST = {}
    SLOTS_LIST['all'] = ALL_SLOTS
    SLOTS_LIST['train'] = slot_train
    SLOTS_LIST['dev'] = slot_dev
    SLOTS_LIST['test'] = slot_test

    return lang, mem_lang, domain_lang, slot_lang, SLOTS_LIST, max_len_val, data_info_dict

