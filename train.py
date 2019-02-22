import os
import tensorflow as tf
import datetime
import getopt
import math
import time
import numpy as np
from model import model_2c_base
from model.model_2c_ensemble import EnsembleModel
from sklearn.metrics import f1_score
from branch_stdout import *
from dataset_ss import DataSet
import relations_builder_2c as rb
import pickle


default_config = {
    "learning_rate": 0.001,
    "display_step": 60,
    "test_step": 300,
    "batch_size": 24,
    "test_batch_size": 600,
    "module_dir": 'Demo/DataSets/s{s}_{v}',
    "style": None,
    "version": "1",
    "k_flod": 1,
    "max_step": 3000,
}


def run(config, ds):
    learning_rate = config['learning_rate']
    display_step = config['display_step']
    test_step = config['test_step']
    batch_size = config['batch_size']
    test_batch_size = config['test_batch_size']
    module_dir = config['module_dir']
    style = config['style']
    version = config['version']
    k_flod = config['k_flod']
    max_step = config['max_step']
    
    module_dir = module_dir.replace("{s}", style + "_k" + str(k_flod)).replace("{v}", "v" + version)
    if not os.path.isdir(module_dir):
        os.mkdir(module_dir)
    cp_path = module_dir

    log_file = module_dir + '/train_log_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.log'
    sys.stdout = BranchStdout(log_file)

    reversed_relation_dict = dict(map(lambda x: (x[1], x[0]), ds.relation_dict.items()))

    m = EnsembleModel(max_train_seq_len=ds.max_train_seq_len,
                      max_test_seq_len=ds.max_test_seq_len,
                      max_train_rel_count=ds.max_train_rel_count,
                      max_test_rel_count=ds.max_test_rel_count)

    saver_best = []
    for i in range(k_flod):
        model_name = "model_" + str(i + 1) + "_" + style
        cm = model_2c_base.Model_board(max_train_seq_len=ds.max_train_seq_len,
                                       max_test_seq_len=ds.max_test_seq_len,
                                       max_train_rel_count=ds.max_train_rel_count,
                                       max_test_rel_count=ds.max_test_rel_count,
                                       ensemble_model=m,
                                       style=style,
                                       name=model_name)
        m.children_models.append(cm)
        sc_file = os.path.join(cp_path, "model_" + str(i + 1) + ".sc")
        if os.path.isfile(sc_file):
            with open(sc_file, 'rb') as f:
                sc_obj = pickle.load(f)
                sc_val = sc_obj[0]
        else:
            sc_val = 0.0
        saver_best.append([tf.train.Saver(var_list=cm.get_saving_variables()),
                           os.path.join(cp_path, model_name + "_best"),
                           sc_val])

    with m.graph.as_default():
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        training_ops = [[optimizer.minimize(m.children_models[j].cost,
                                            var_list=m.children_models[j].get_trainable_variables())
                         for j in range(k_flod) if j != i] + [m.get_train_update_ops()]
                        for i in range(k_flod)]
        watching_ops = [[(m.children_models[j].train_cost, m.children_models[j].regularization_cost,
                          m.children_models[j].cost, m.children_models[j].pred)
                         for j in range(k_flod) if j != i]
                        for i in range(k_flod)]
        message_template = ['\n'.join(["M" + str(j + 1) + ", Minibatch Loss={" +
                                       str(j * 4 if j < i else (j - 1) * 4) + ":f}/{" +
                                       str((j * 4 if j < i else (j - 1) * 4) + 1) + ":f}/{" +
                                       str((j * 4 if j < i else (j - 1) * 4) + 2) + ":f}, F1={" +
                                       str((j * 4 if j < i else (j - 1) * 4) + 3) + ":f}"
                                       for j in range(k_flod) if j != i])
                            for i in range(k_flod)]
        test_watching_ops = [(m.children_models[j].test_cost,
                              m.children_models[j].test_pred,
                              m.children_models[j].test_score)
                             for j in range(k_flod)]

        with tf.Session(graph=m.graph) as sess:
            tf.set_random_seed(20140630)
            sess.run(tf.global_variables_initializer())

            print("learning_rate=", learning_rate)
            print("batch_size=", batch_size)
            print("seq_mask_rate", ds.seq_mask_rate)
            print("max_rel_distance=", ds.max_rel_distance)
            print("min_rel_distance=", ds.min_rel_distance)
            print("split_size=", ds.split_size)

            step = 0
            ds.reset_train()
            epoch_steps = math.ceil(ds.train_count / batch_size)
            sess_time_in_display = 0
            valid_labels = [1]

            while max_step > 0 and step < max_step:
                es = step % epoch_steps
                step += 1

                batch_data_seq, batch_data_len, \
                batch_data_rel, batch_data_rsl, batch_doc, batch_len = \
                    ds.next_train(batch_size=batch_size)
                batch_data_lbl = np.cast[np.float32](batch_data_rel[:, :, 0] > 0)

                if step % display_step == 0:
                    st = time.time()
                    w_v, _ = \
                        sess.run([watching_ops[es % k_flod], training_ops[es % k_flod]],
                                 feed_dict={
                                     m.xci: batch_data_seq[:, :, 0],
                                     m.xtc: batch_data_seq[:, :, 1],
                                     m.xsl: batch_data_len,
                                     m.rel: batch_data_rel[:, :, 1:3],
                                     m.lbl: batch_data_lbl,
                                     m.rsl: batch_data_rsl
                                 })
                    sess_time_in_step = time.time() - st
                    sess_time_in_display += sess_time_in_step
                    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "Iter " + str(step))
                    print(message_template[es % k_flod].format(
                        *[v for sl in
                          [[w_v[i][0], w_v[i][1], w_v[i][2],
                            f1_score(batch_data_lbl.reshape([-1]), w_v[i][3].reshape([-1]),
                                     labels=valid_labels,
                                     average="micro")] for i in range(k_flod - 1)]
                          for v in sl]
                    ))
                    print("Sess Time={0:f}/{1:f}".format(sess_time_in_step, sess_time_in_display))
                    sess_time_in_display = 0
                    sys.stdout.flush()
                else:
                    st = time.time()
                    sess.run(training_ops[es % k_flod], feed_dict={m.xci: batch_data_seq[:, :, 0],
                                                                   m.xtc: batch_data_seq[:, :, 1],
                                                                   m.xsl: batch_data_len,
                                                                   m.rel: batch_data_rel[:, :, 1:3],
                                                                   m.lbl: batch_data_lbl,
                                                                   m.rsl: batch_data_rsl})
                    sess_time_in_display += time.time() - st

                if step % test_step == 0:
                    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "Iter " + str(step))
                    for i in range(k_flod):
                        epoch_test_stat = [0.0, 0.0, 0.0, 0.0]
                        epoch_test_count = 0

                        all_data_seq, all_data_len, \
                        all_data_rel, all_data_rsl, all_doc, all_entities, all_len = \
                            ds.next_flod_test(i, k_flod, batch_size)
                        all_data_lbl = np.cast[np.float32](all_data_rel[:, :, 0] > 0)

                        j = 0
                        all_pred_score = np.zeros((all_len, ds.all_regular_rel_count), dtype=np.float32)
                        while j < len(all_data_seq):
                            batch_data_seq = all_data_seq[j:j + test_batch_size]
                            batch_data_len = all_data_len[j:j + test_batch_size]
                            batch_data_rel = all_data_rel[j:j + test_batch_size]
                            batch_data_rsl = all_data_rsl[j:j + test_batch_size]
                            batch_data_lbl = all_data_lbl[j:j + test_batch_size]

                            w_v = \
                                sess.run(test_watching_ops,
                                         feed_dict={
                                             m.test_xci: batch_data_seq[:, :, 0],
                                             m.test_xtc: batch_data_seq[:, :, 1],
                                             m.test_xsl: batch_data_len,
                                             m.test_rel: batch_data_rel[:, :, 1:3],
                                             m.test_lbl: batch_data_lbl,
                                             m.test_rsl: batch_data_rsl
                                         })

                            epoch_test_stat[0] += f1_score(batch_data_lbl.reshape([-1]), w_v[i][1].reshape([-1]),
                                                           labels=valid_labels,
                                                           average="micro")
                            epoch_test_stat[1] += w_v[i][0]
                            epoch_test_count += 1
                            if j + test_batch_size <= len(all_data_seq):
                                all_pred_score[j:j + test_batch_size, :] = w_v[i][2]
                            else:
                                all_pred_score[j:, :] = w_v[i][2]
                            j += test_batch_size

                        raw_relations_p = rb.build_raw_relations(rb.build_relations_from_data(
                            doc=all_doc,
                            filenames=ds.train_filenames,
                            entities=all_entities,
                            rel=all_data_rel,
                            rsl=all_data_rsl,
                            pred_score=all_pred_score,
                            relation_labels=reversed_relation_dict))

                        raw_relations_f = ds.get_raw_relations_of_flod_k(i, k_flod, batch_size)
                        cr = rb.compare_result(raw_relations_p, raw_relations_f)
                        model_score = cr['micro']['F1']

                        print("M" + str(i + 1),
                              ", Test Loss={0:f}".format(epoch_test_stat[1] / epoch_test_count),
                              ", F1={0:f}".format(epoch_test_stat[0] / epoch_test_count),
                              ", Micro[F1={0:f}|P={1:f}|R={2:f}]".format(cr['micro']['F1'],
                                                                         cr['micro']['P'],
                                                                         cr['micro']['R']),
                              ", Macro[F1={0:f}|P={1:f}|R={2:f}]".format(cr['macro']['F1'],
                                                                         cr['macro']['P'],
                                                                         cr['macro']['R']))

                        if model_score > saver_best[i][2]:
                            saver_best[i][0].save(sess, saver_best[i][1])
                            sc_file = os.path.join(cp_path, "model_" + str(i + 1) + ".sc")
                            with open(sc_file, 'wb') as f:
                                pickle.dump((model_score, cr['micro'], cr['macro']), f)
                            saver_best[i][2] = model_score
                            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                  'Got best model of model', str(i + 1), '-', model_score)

                    sys.stdout.flush()


if __name__ == "__main__":
    stdout = sys.stdout
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "Run Start")
    config = default_config.copy()
    data_path = 'Demo/DataSets'

    try:
        opts, args = getopt.getopt(sys.argv[1:], "d:r:s:v:k:",
                                   ["dpath=", "mdir=", "style=", "version=", "k-flod=", "max-step="])
    except getopt.GetoptError:
        print('arguments error')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-d", "--dpath"):
            data_path = arg
        elif opt in ("-r", "--mdir"):
            config['module_dir'] = arg
        elif opt in ("-v", "--version"):
            config['version'] = arg
        elif opt in ("-k", "--k-flod"):
            config['k_flod'] = int(arg)
        elif opt in ("--max-step",):
            config['max_step'] = int(arg)

    np.random.seed(20140630)
    ts = DataSet(os.path.join(data_path, 'ruijin_round2_train/ruijin_round2_train'),
                 test_size=0,
                 seq_mask_rate=0.05,
                 min_rel_distance=0,
                 max_rel_distance=70,
                 max_split_rel_count=139,
                 all_regular_rel_count=260,
                 split_size=140,
                 worker_count=4,
                 capacity=240,
                 return_entity_detail_id=False)

    config["style"] = "1"
    run(config, ts)
    sys.stdout = stdout
    config["style"] = "6"
    run(config, ts)
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "Finished")


