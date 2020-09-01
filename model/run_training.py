from tqdm import tqdm
from model.predict import predict

def run_epoch(ep, total_loss, state_out, train_op, global_step, train_summaries, losses,
              nb_tokens, sess, src_lang, num_batches, summary_writer, args, domain_lang=None,
              slot_lang=None, evaluation_variable=None, evaluator=None, is_eval=False):

    avg_lenval_loss = 0
    avg_gate_loss = 0
    avg_state_loss = 0

    epoch_lenval_loss = 0
    epoch_gate_loss = 0
    epoch_state_loss = 0

    avg_slot_nb_tokens = 0
    avg_state_nb_tokens = 0
    avg_gate_nb_tokens = 0

    epoch_slot_nb_tokens = 0
    epoch_state_nb_tokens = 0
    epoch_gate_nb_tokens = 0

    epoch_joint_lenval_matches = 0
    epoch_joint_gate_matches = 0
    total_samples = 0

    if args['pointer_decoder']:
        predict_lang = src_lang

    if is_eval:
        predictions = {}

    for i in tqdm(range(0, num_batches)):
        _, _gs, _summary, _losses, _nb_tokens, _state_out, _evaluation_variable = sess.run([train_op,
                                                                                           global_step,
                                                                                           train_summaries,
                                                                                           losses,
                                                                                           nb_tokens,
                                                                                           state_out,
                                                                                           evaluation_variable])
        summary_writer.add_summary(_summary, _gs)

        if is_eval:
            matches, predictions = predict(_state_out, _evaluation_variable, predict_lang, domain_lang, slot_lang, predictions, True, src_lang, args)
            epoch_joint_lenval_matches += matches['joint_lenval']
            epoch_joint_gate_matches += matches['joint_gate']
            total_samples += len(_evaluation_variable['turn_id'])

        avg_lenval_loss += _losses['lenval_loss']
        avg_gate_loss += _losses['gate_loss']
        avg_state_loss += _losses['state_loss']

        avg_gate_nb_tokens += _nb_tokens['gate']
        avg_slot_nb_tokens += _nb_tokens['slot']
        avg_state_nb_tokens += _nb_tokens['state']

        epoch_slot_nb_tokens += _nb_tokens['slot']
        epoch_state_nb_tokens += _nb_tokens['state']
        epoch_gate_nb_tokens += _nb_tokens['gate']

        epoch_lenval_loss += _losses['lenval_loss']
        epoch_state_loss += _losses['state_loss']
        epoch_gate_loss += _losses['gate_loss']

        if (i+1) % args['reportp'] == 0 and not is_eval:
            avg_lenval_loss /= avg_slot_nb_tokens
            avg_state_loss /= avg_state_nb_tokens
            avg_gate_loss /= avg_gate_nb_tokens
            print("Step {} gate loss {:.4f} lenval loss {:.4f} state loss {:.4f}".
                format(i+1, avg_gate_loss, avg_lenval_loss, avg_state_loss))
            with open(args['path'] + '/train_log.csv', 'a') as f:
                f.write('{},{},{},{},{}\n'.format(ep+1, i+1, avg_gate_loss, avg_lenval_loss, avg_state_loss))
            avg_lenval_loss = 0
            avg_slot_nb_tokens = 0
            avg_state_loss = 0
            avg_state_nb_tokens = 0
            avg_gate_loss = 0
            avg_gate_nb_tokens = 0

    epoch_lenval_loss /= epoch_slot_nb_tokens
    epoch_state_loss /= epoch_state_nb_tokens
    epoch_gate_loss /= epoch_gate_nb_tokens
    joint_gate_acc, joint_lenval_acc, joint_acc_score, F1_score, turn_acc_score = 0, 0, 0, 0, 0

    real_joint_acc_score = 0.0


    if is_eval:
        joint_lenval_acc = 1.0 * epoch_joint_lenval_matches/total_samples
        joint_gate_acc = 1.0 * epoch_joint_gate_matches/total_samples 
        joint_acc_score, F1_score, turn_acc_score = -1, -1, -1
        #join accuracy score, turn accuracy score, F1 score 구하기
        joint_acc_score, F1_score, turn_acc_score = evaluator.evaluate_metrics(predictions, 'dev')


    print(
        "Epoch {} gate loss {:.4f} lenval loss {:.4f} state loss {:.4f} \n joint_gate acc {:.4f} joint_lenval acc {:.4f} joint acc {:.4f} f1 {:.4f} turn acc {:.4f}".
        format(ep + 1, epoch_gate_loss, epoch_lenval_loss, epoch_state_loss,
               joint_gate_acc, joint_lenval_acc, joint_acc_score, F1_score, turn_acc_score))
    print(args['path'])
    with open(args['path'] + '/val_log.csv', 'a') as f:
        if is_eval:
            split = 'dev'
        else:
            split = 'train'
        f.write('{},{},{},{},{},{},{},{},{},{}\n'.
                format(ep + 1, split,
                       epoch_gate_loss, epoch_lenval_loss, epoch_state_loss,
                       joint_gate_acc, joint_lenval_acc,
                       joint_acc_score, F1_score, turn_acc_score))
    if is_eval:
        return (epoch_gate_loss + epoch_lenval_loss + epoch_state_loss) / 3, (joint_gate_acc + joint_lenval_acc + joint_acc_score) / 3, joint_acc_score
    else:
        return _gs



