import tensorflow as tf
import datetime
import os
from dataset_ss import DataSet
from model.model_2c_ensemble import EnsembleModel
from model import model_2c_base
import relations_builder_2c as rb


def run(models, ds, output_dir):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    with tf.Graph().as_default() as g:
        m = EnsembleModel(max_test_seq_len=ds.max_test_seq_len,
                          max_test_rel_count=ds.max_test_rel_count,
                          graph=g)

        for mf in models:
            for mn in models[mf]:
                n_arr = mn.split('@')
                if len(n_arr) > 1:
                    style = n_arr[-1]
                    mn = n_arr[0]
                else:
                    style = mn.split('_')[-1]

                m.children_models.append(model_2c_base.Model_board(max_test_seq_len=ds.max_test_seq_len,
                                                                   max_test_rel_count=ds.max_test_rel_count,
                                                                   ensemble_model=m,
                                                                   style=style,
                                                                   name=mn))

        score = [cm.test_score for cm in m.children_models]

        saver_copy = {}
        for mf in models:
            # 收集复制源图中的变量
            var_names_in_copy_file = []
            with tf.Graph().as_default():
                tf.train.import_meta_graph(mf + ".meta")
                for tv in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                    var_names_in_copy_file.append(tv.op.name)

            # 对比当前模型中的变量，同名的是要加入到copy表中
            copy_variables = {}
            for tvc in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                var_name = tvc.op.name

                if var_name in var_names_in_copy_file:
                    found = False
                    for prefix in models[mf]:
                        if var_name.startswith(prefix + '/'):
                            found = True
                            break
                    if found:
                        copy_variables[var_name] = tvc

            print('copy {0} variables from {1}'.format(len(copy_variables), mf))
            if len(copy_variables) > 0:
                saver_copy[mf] = tf.train.Saver(copy_variables)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for mf in models:
                if mf in saver_copy:
                    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'copy from', mf)
                    saver_copy[mf].restore(sess, mf)

            reversed_relation_dict = dict(map(lambda t: (t[1], t[0]), ds.relation_dict.items()))

            handled_count = 0
            while not ds.is_test_end:
                batch_data_seq, batch_data_len, \
                batch_data_rel, batch_data_rsl, batch_doc, batch_entities, batch_len = \
                    ds.next_exec(batch_size=4)

                result_scores = sess.run(score, feed_dict={m.test_xci: batch_data_seq[:, :, 0],
                                                           m.test_xtc: batch_data_seq[:, :, 1],
                                                           m.test_xsl: batch_data_len,
                                                           m.test_rel: batch_data_rel[:, :, 1:3],
                                                           m.test_rsl: batch_data_rsl})

                relations = []
                for i in range(len(m.children_models)):
                    relations.extend(rb.build_relations_from_data(doc=batch_doc,
                                                                  filenames=ds.test_filenames,
                                                                  entities=batch_entities,
                                                                  rel=batch_data_rel,
                                                                  rsl=batch_data_rsl,
                                                                  pred_score=result_scores[i],
                                                                  relation_labels=reversed_relation_dict))
                raw_relations = rb.build_raw_relations(rb.merge_relations(relations))
                rb.to_files(raw_relations, output_dir)

                handled_count += len(raw_relations)
                print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      'processing %d/%d' % (handled_count, ds.test_count))


if __name__ == "__main__":
    data_path = 'Demo/DataSets/ruijin_round2_test_b/ruijin_round2_test_b'
    module_files = {
        'Demo/DataSets/s1_k5_v1/model_1_1_best': ["model_1_1"],
        'Demo/DataSets/s1_k5_v1/model_2_1_best': ["model_2_1"],
        'Demo/DataSets/s1_k5_v1/model_3_1_best': ["model_3_1"],
        'Demo/DataSets/s1_k5_v1/model_4_1_best': ["model_4_1"],
        'Demo/DataSets/s1_k5_v1/model_5_1_best': ["model_5_1"],
        'Demo/DataSets/s6_k5_v1/model_1_6_best': ["model_1_6"],
        'Demo/DataSets/s6_k5_v1/model_2_6_best': ["model_2_6"],
        'Demo/DataSets/s6_k5_v1/model_3_6_best': ["model_3_6"],
        'Demo/DataSets/s6_k5_v1/model_4_6_best': ["model_4_6"],
        'Demo/DataSets/s6_k5_v1/model_5_6_best': ["model_5_6"]
    }
    output_dir = 'submit'

    ts = DataSet(data_path,
                 max_rel_distance=70,
                 split_size=140,
                 all_regular_rel_count=300,
                 shuffle=False,
                 test_size=1,
                 worker_count=0,
                 enabled_pre_build=False)

    run(module_files, ts, output_dir)
