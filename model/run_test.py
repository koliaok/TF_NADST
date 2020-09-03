from tqdm import tqdm
from model.predict import predict

import copy
import json
import pickle as pkl

def make_test_data_set(start, end, test_data_info, model=None, slot_gating=None):
    test_data= {
        "ID": test_data_info['ID'][start:end],
        "turn_id": test_data_info['turn_id'][start:end],
        "context": test_data_info['context'][start:end],
        "delex_context_plain": test_data_info['delex_context_plain'][start:end],
        "delex_context": test_data_info['delex_context'][start:end],
        "context_plain": test_data_info['context_plain'][start:end],
        "sorted_in_domains": test_data_info['sorted_in_domains'][start:end],
        "turn_belief": test_data_info['turn_belief'][start:end],
        "sorted_in_domains2": test_data_info['sorted_in_domains2'][start:end],
        "sorted_in_slots": test_data_info['sorted_in_slots'][start:end],
        "sorted_in_slots2": test_data_info['sorted_in_slots2'][start:end],
        "sorted_lenval": test_data_info['sorted_lenval'][start:end],
        "sorted_generate_y": test_data_info['sorted_generate_y'][start:end],
        "context_mask": test_data_info['context_mask'][start:end],
        "delex_context_mask": test_data_info['delex_context_mask'][start:end],
        "sorted_in_domainslots_mask": test_data_info['sorted_in_domainslots_mask'][start:end],
    }
    feed_dict = make_feed_dict(model, test_data)
    if slot_gating:
        feed_dict[model.sorted_gates] = test_data_info["sorted_gates"][start:end]
    return test_data, feed_dict

def make_feed_dict(model, test_data):
    feed_dict = {
        model.contexts: test_data["context"],
        model.delex_contexts: test_data["delex_context"],
        model.sorted_in_domainss: test_data["sorted_in_domains"],
        model.sorted_in_slotss: test_data["sorted_in_slots"],
        model.sorted_in_domains2s: test_data["sorted_in_domains2"],
        model.sorted_in_slots2s: test_data["sorted_in_slots2"],
        model.sorted_generate_ys: test_data["sorted_generate_y"],
        model.sorted_lenvals: test_data["sorted_lenval"],
    }
    return feed_dict

        
    

def run_test(
             src_lang, num_batches, args, domain_lang=None,
             slot_lang=None, model=None,
             evaluator=None, all_slot_list=None, test_data=None,
             is_eval=False):

    predictions = {}
    latencies = []
    src_lens = []
    tgt_lens = []
    oracle_predictions = {}
    joint_gate_matches = 0
    joint_lenval_matches = 0
    total_samples = 0

    if args['pointer_decoder']:
        predict_lang = src_lang

    if is_eval:
        predictions = {}

    for i in tqdm(range(0, num_batches)):
        start_range = i*args["eval_batch"]
        end_range = (i+1)*args["eval_batch"]
        batch_data, feed_dict = make_test_data_set(start_range, end_range, test_data, model=model, slot_gating=args['slot_gating'])


        _gs, _losses, _nb_tokens, _state_out, _evaluation_variable = model.sess.run([model.global_step,
                                                                                    model.losses,
                                                                                    model.nb_tokens,
                                                                                    model.state_out,
                                                                                    model.evaluation_variable], feed_dict=feed_dict)


        if is_eval:
            predictions, latencies, src_lens, tgt_lens = predict(_state_out, _evaluation_variable, predict_lang,
                                                                 domain_lang, slot_lang, predictions, False, src_lang, args,
                                                                 feed_dict_func = make_feed_dict, batch_data=batch_data, model=model, slot_list=all_slot_list,
                                                                 latency=latencies, src_lens=src_lens, tgt_lens=tgt_lens, test=True)

            matches, oracle_predictions = predict(_state_out, _evaluation_variable, predict_lang, domain_lang, slot_lang,
                                                  oracle_predictions, True, src_lang, args, batch_data=batch_data, test=True)

            joint_lenval_matches += matches['joint_lenval']
            joint_gate_matches += matches['joint_gate']
            total_samples += len(batch_data['turn_id'])

    avg_latencies = sum(latencies) / len(latencies)
    print("Average latency: {}".format(avg_latencies))
    with open(args['path'] + '/latency_eval.csv', 'w') as f:
        f.write(str(avg_latencies))
    pkl.dump(zip(latencies, src_lens, tgt_lens), open(args['path'] + '/latency_out.pkl', 'wb'))
    joint_acc_score, F1_score, turn_acc_score = -1, -1, -1
    oracle_joint_acc, oracle_f1, oracle_acc = -1, -1, -1
    joint_acc_score, F1_score, turn_acc_score = evaluator.evaluate_metrics(predictions, 'test')
    oracle_joint_acc, oracle_f1, oracle_acc = evaluator.evaluate_metrics(oracle_predictions, 'test')
    joint_lenval_acc = 1.0 * joint_lenval_matches / total_samples
    joint_gate_acc = 1.0 * joint_gate_matches / total_samples
    with open(args['path'] + '/eval_{}_epoch{}_ptest{}-{}.csv'.format(args['test_split'], args['eval_epoch'],
                                                                      args['p_test'], args['p_test_fertility']),
              'a') as f:
        f.write("{},{},{},{},{},{},{},{}".
                format(joint_gate_acc, joint_lenval_acc,
                       joint_acc_score, turn_acc_score, F1_score,
                       oracle_joint_acc, oracle_acc, oracle_f1))
    print("Joint Gate Acc {}".format(joint_gate_acc))
    print("Joint Lenval Acc {}".format(joint_lenval_acc))
    print("Joint Acc {} Slot Acc {} F1 {}".format(joint_acc_score, turn_acc_score, F1_score))
    print("Oracle Joint Acc {} Slot Acc {} F1 {}".format(oracle_joint_acc, oracle_f1, oracle_acc))
    json.dump(predictions, open(
        args['path'] + '/predictions_{}_epoch{}_ptest{}-{}.json'.format(args['test_split'], args['eval_epoch'],
                                                                        args['p_test'], args['p_test_fertility']), 'w'),
              indent=4)
    json.dump(oracle_predictions, open(
        args['path'] + '/oracle_predictions_{}_epoch{}_ptest{}-{}.json'.format(args['test_split'], args['eval_epoch'],
                                                                               args['p_test'],
                                                                               args['p_test_fertility']), 'w'), indent=4)



