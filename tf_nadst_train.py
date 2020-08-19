from utils_s.utils_multiWOZ_DST import *
from utils_s.config import *
from utils_s.utils import *
from model.nadst import NADST
from model.run_training import run_epoch
from model.evaluator import Evaluator

import tensorflow as tf
import pdb
import os
import os.path
import pickle as pkl
import logging

EAGER_EXCUTION = False

if EAGER_EXCUTION:
    tf.compat.v1.enable_eager_execution()
else:
    tf.compat.v1.disable_eager_execution()

epoch = 200
logging.basicConfig(level=logging.INFO)

cnt = 0.0
min_dev_loss = float("Inf")
max_dev_acc = -float("Inf")
max_dev_slot_acc = -float("Inf")

if not os.path.exists(args['path']):
    os.makedirs(args['path'])
if not os.path.exists(args['save_path']):
    os.makedirs(args['save_path'])

if not os.path.exists(args['path']+'/data.pkl'):
    src_lang, tgt_lang, domain_lang, slot_lang, SLOTS_LIST, max_len_val, data_info_dic = prepare_data_seq(True, args)

    save_data = {
        'src_lang': src_lang,
        'tgt_lang': tgt_lang,
        'domain_lang': domain_lang,
        'slot_lang': slot_lang,
        'SLOTS_LIST': SLOTS_LIST,
        'data_info_dic': data_info_dic,
        'max_len_val': max_len_val}

    pkl.dump(save_data, open(args['path'] + '/data.pkl', 'wb'))

else:
    with open(args['path']+'/data.pkl', 'rb') as data:
        save_data = pkl.load(data)

src_lang = save_data["src_lang"]
tgt_lang = save_data["tgt_lang"]
domain_lang = save_data["domain_lang"]
slot_lang = save_data["slot_lang"]
SLOTS_LIST = save_data["SLOTS_LIST"]
args = args
data_info_dic = save_data["data_info_dic"]
max_len_val = save_data["max_len_val"]
eval_batch = args["eval_batch"] if args["eval_batch"] else args["batch"]

#training tensor data setup
if EAGER_EXCUTION:
    xs, ys = get_eager_tensor_dataset(data_info_dic["train"], args['batch'], args['slot_gating'])

else:
    train_dataset, train_num_batches, train_num_sample = get_tensor_dataset(data_info_dic["train"], args['batch'], args['slot_gating'], True)
    eval_dataset, eval_num_batches, eval_num_sample = get_tensor_dataset(data_info_dic["dev"], eval_batch, args['slot_gating'], False)

    iter = tf.compat.v1.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    xs, ys = iter.get_next()

    train_init_op = iter.make_initializer(train_dataset)
    eval_init_op = iter.make_initializer(eval_dataset)

nadst = NADST()
total_loss, train_op, global_step, train_summaries, losses,\
nb_tokens, state_out, evaluation_variable = nadst.model(xs=xs, ys=ys, src_lang=src_lang,
                                                        domain_lang=domain_lang, slot_lang=slot_lang,
                                                        len_val=max_len_val, args=args, training=True)

eval_total_loss, eval_train_op, eval_global_step, eval_train_summaries, eval_losses,\
eval_nb_tokens, eval_state_out, eval_evaluation_variable = nadst.model(xs=xs, ys=ys, src_lang=src_lang,
                                                        domain_lang=domain_lang, slot_lang=slot_lang,
                                                        len_val=max_len_val, args=args, training=False)



logging.info("# Load model complete")
#
evaluator = Evaluator(SLOTS_LIST)

#start training
logging.info("# Open Tensor Session")
saver = tf.compat.v1.train.Saver(max_to_keep=5)

with open(args['path'] + '/train_log.csv', 'w') as f:
    f.write('epoch,step,gate_loss,lenval_loss,state_loss\n')
with open(args['path'] + '/val_log.csv', 'w') as f:
    f.write('epoch,split,gate_loss,lenval_loss,state_loss,joint_gate_acc,joint_lenval_acc,joint_acc,f1,turn_acc\n')
json.dump(args, open(args['path'] + '/params.json', 'w'))

with tf.compat.v1.Session() as sess:
    ckpt = tf.compat.v1.train.latest_checkpoint(args['save_path'])
    if ckpt is None:
        logging.info("Initializing from scratch")
        sess.run(tf.compat.v1.global_variables_initializer())
        save_variable_specs(os.path.join(args['save_path'], "specs"))
    else:
        saver.restore(sess, ckpt)
    summary_writer = tf.compat.v1.summary.FileWriter(args['save_path'], sess.graph)

    sess.run(train_init_op)
    total_steps = epoch * train_num_batches
    _gs = sess.run(global_step)
    run_operation = (train_init_op, eval_init_op)

    for ep in range(epoch):

        _gs = run_epoch(ep, total_loss, state_out, train_op, global_step, train_summaries, losses, nb_tokens,
                        sess, src_lang, train_num_batches, summary_writer, args, evaluation_variable=evaluation_variable, is_eval=False)

        modelfile = args['save_path'] + '/nadst_model_epoch{}'.format(ep + 1)
        if ((ep + 1) % int(args['evalp']) == 0):
        #if True:
            _ = sess.run(eval_init_op)
            dev_loss, dev_acc, dev_joint_acc = run_epoch(ep, total_loss, state_out, train_op, global_step, train_summaries,
                                                          losses, nb_tokens, sess, src_lang, eval_num_batches, summary_writer,
                                                          args, domain_lang=domain_lang, slot_lang=slot_lang, evaluation_variable=evaluation_variable,
                                                         evaluator=evaluator, is_eval=True)


            """
            dev_loss, dev_acc, dev_joint_acc = run_epoch(ep, eval_total_loss, eval_state_out, eval_train_op, eval_global_step, eval_train_summaries,
                                                         eval_losses, eval_nb_tokens, sess, src_lang, eval_num_batches,
                                                         summary_writer, args, domain_lang=domain_lang, slot_lang=slot_lang,
                                                         evaluation_variable=eval_evaluation_variable,
                                                         evaluator=evaluator, is_eval=True)
            """

            print('deve loss is {}'.format(dev_acc))
            if args['eval_metric'] == 'acc':
                check = (dev_acc > max_dev_acc)
            elif args['eval_metric'] == 'slot_acc':
                check = (dev_joint_acc > max_dev_slot_acc)
            elif args['eval_metric'] == 'loss':
                check = (dev_loss < min_dev_loss)
            if check:
                logging.info("# save models")
                saver.save(sess, modelfile)
                logging.info("after training of {} epochs, {} has been saved.".format(epoch, modelfile))
                cnt = 0
                best_model_id = ep + 1
                print('Dev loss changes from {} --> {}'.format(min_dev_loss, dev_loss))
                print('Dev acc changes from {} --> {}'.format(max_dev_acc, dev_acc))
                print('Dev slot acc changes from {} --> {}'.format(max_dev_slot_acc, dev_joint_acc))
                min_dev_loss = dev_loss
                max_dev_acc = dev_acc
                max_dev_slot_acc = dev_joint_acc
            else:
                cnt += 1
            if (cnt == args["patience"]):
                print("Ran out of patient, early stop...")
                break
            sess.run(train_init_op)
    print("The best model is at epoch {}".format(best_model_id))


            #save model



