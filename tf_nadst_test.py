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

epoch = args['eval_epoch']
logging.basicConfig(level=logging.INFO)


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
    xs, ys = get_eager_tensor_dataset(data_info_dic["test"], args['batch'])
else:
    test_dataset, test_num_batches, test_num_sample = get_tensor_dataset(data_info_dic["test"], eval_batch, False)

    iter = tf.compat.v1.data.Iterator.from_structure(test_dataset.output_types, test_dataset.output_shapes)
    xs, ys = iter.get_next()
    test_init_op = iter.make_initializer(test_dataset)

nadst = NADST()
eval_total_loss, evel_losses, evel_nb_tokens, eval_state_out, \
eval_evaluation_variable = nadst.model(xs=xs, ys=ys, src_lang=src_lang,
                                        domain_lang=domain_lang, slot_lang=slot_lang,
                                        len_val=max_len_val, args=args, training=False)
logging.info("# Load model complete")
#
evaluator = Evaluator(SLOTS_LIST)

#start training
logging.info("# Open Tensor Session")
saver = tf.compat.v1.train.Saver(max_to_keep=epoch)

with open(args['path'] + '/eval_{}_epoch{}_ptest{}-{}.csv'.format(args['test_split'], args['eval_epoch'], args['p_test'], args['p_test_fertility']), 'w') as f:
    f.write('joint_lenval_acc,joint_acc,slot_acc,f1,oracle_joint_acc,oracle_slot_acc,oracle_f1\n')


with tf.compat.v1.Session() as sess:
    ckpt = tf.compat.v1.train.latest_checkpoint(args['save_path'])
    if ckpt is None:
        logging.info("Initializing from scratch")
        sess.run(tf.compat.v1.global_variables_initializer())
        save_variable_specs(os.path.join(args['save_path'], "specs"))
    else:
        saver.restore(sess, ckpt)
    summary_writer = tf.compat.v1.summary.FileWriter(args['save_path'], sess.graph)

    sess.run(test_init_op)
    total_steps = epoch * test_num_batches
    _gs = sess.run(global_step)
    run_operation = (test_init_op)

    for ep in range(epoch):
        dev_loss, dev_acc, dev_joint_acc = run_epoch(ep, total_loss, state_out, train_op, global_step, train_summaries,
                                                      losses, nb_tokens, sess, src_lang, test_num_batches, summary_writer,
                                                      args, domain_lang=domain_lang, slot_lang=slot_lang, evaluation_variable=evaluation_variable,
                                                     evaluator=evaluator, is_eval=True)



