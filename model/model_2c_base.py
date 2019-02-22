import tensorflow as tf
import abc


class Model(object, metaclass=abc.ABCMeta):
    def __init__(self,
                 max_train_seq_len=1500,
                 max_test_seq_len=1500,
                 max_train_rel_count=1500,
                 max_test_rel_count=1500,
                 is_weight_regularization=True,
                 ensemble_model=None,
                 graph=None,
                 name="model"):
        self.name = name

        if ensemble_model is not None:
            self.graph = ensemble_model.graph
        else:
            if graph is None:
                graph = tf.Graph()
            self.graph = graph

        with self.graph.as_default():
            with tf.variable_scope(name):
                if ensemble_model is None:
                    self.xci = tf.placeholder(dtype=tf.int32, shape=[None, max_train_seq_len])
                    self.xtc = tf.placeholder(dtype=tf.int32, shape=[None, max_train_seq_len])
                    self.xsl = tf.placeholder(dtype=tf.int32, shape=[None])
                    self.rel = tf.placeholder(dtype=tf.int32, shape=[None, max_train_rel_count, 2])
                    self.lbl = tf.placeholder(dtype=tf.float32, shape=[None, max_train_rel_count])
                    self.rsl = tf.placeholder(dtype=tf.int32, shape=[None])

                    self.test_xci = tf.placeholder(dtype=tf.int32, shape=[None, max_test_seq_len])
                    self.test_xtc = tf.placeholder(dtype=tf.int32, shape=[None, max_test_seq_len])
                    self.test_xsl = tf.placeholder(dtype=tf.int32, shape=[None])
                    self.test_rel = tf.placeholder(dtype=tf.int32, shape=[None, max_test_rel_count, 2])
                    self.test_lbl = tf.placeholder(dtype=tf.float32, shape=[None, max_test_rel_count])
                    self.test_rsl = tf.placeholder(dtype=tf.int32, shape=[None])
                else:
                    self.xci = ensemble_model.xci
                    self.xtc = ensemble_model.xtc
                    self.xsl = ensemble_model.xsl
                    self.rel = ensemble_model.rel
                    self.lbl = ensemble_model.lbl
                    self.rsl = ensemble_model.rsl

                    self.test_xci = ensemble_model.test_xci
                    self.test_xtc = ensemble_model.test_xtc
                    self.test_xsl = ensemble_model.test_xsl
                    self.test_rel = ensemble_model.test_rel
                    self.test_lbl = ensemble_model.test_lbl
                    self.test_rsl = ensemble_model.test_rsl

                with tf.name_scope("train"):
                    self.train_net = self.build_train_net(self.xci, self.xtc, self.xsl, self.rel, self.rsl)
                    out = tf.squeeze(self.train_net.out, axis=-1)

                    step = tf.get_variable("step", shape=(), initializer=tf.zeros_initializer, dtype=tf.int32)
                    self.model_step = step
                    tf.add_to_collection(self.train_net.UPDATE_OPS_COLLECTION, tf.assign_add(step, 1))

                    relation_mask = tf.sequence_mask(self.rsl, max_train_rel_count, dtype=tf.float32)
                    self.pred = tf.nn.relu(tf.sign(out)) * relation_mask

                    total_valid_samples = tf.reduce_sum(relation_mask) + 1e-6

                    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=out,
                                                                   labels=self.lbl)
                    loss = loss * relation_mask

                    # 评价的cost保持不变
                    self.train_cost = tf.reduce_sum(loss) / total_valid_samples

                    if is_weight_regularization:
                        # 得到所有可以训练的参数，即所有trainable=True 的tf.Variable/tf.get_variable
                        tv = tf.trainable_variables()
                        # 0.001是lambda超参数
                        regularization_cost = 0.001 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv
                                                                     if v.name.startswith(name + '/main/')])
                        self.regularization_cost = regularization_cost
                        self.cost = self.train_cost + regularization_cost
                    else:
                        self.cost = self.train_cost

                with tf.name_scope("test"):
                    self.test_net = self.build_test_net(self.test_xci, self.test_xtc, self.test_xsl, self.test_rel, self.test_rsl)
                    out = tf.squeeze(self.test_net.out, axis=-1)

                    relation_mask = tf.sequence_mask(self.test_rsl, max_test_rel_count, dtype=tf.float32)
                    self.test_pred = tf.nn.relu(tf.sign(out)) * relation_mask
                    self.test_score = tf.nn.tanh(out) * relation_mask
                    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=out,
                                                                   labels=self.test_lbl)
                    cost = tf.reduce_sum(loss * relation_mask) / (tf.reduce_sum(relation_mask) + 1e-6)

                    self.test_cost = cost

    def set_model_step(self, value=0.0):
        with self.graph.as_default():
            return tf.assign(self.model_step, value)

    def get_train_update_ops(self):
        with self.graph.as_default():
            return tf.get_collection(self.train_net.UPDATE_OPS_COLLECTION)

    def get_trainable_variables(self):
        with self.graph.as_default():
            return tf.trainable_variables(self.name)

    def get_saving_variables(self):
        with self.graph.as_default():
            return tf.global_variables(self.name)

    @abc.abstractmethod
    def build_train_net(self, xci, xtc, xsl, rel, rsl):
        pass

    @abc.abstractmethod
    def build_test_net(self, xci, xtc, xsl, rel, rsl):
        pass


# from model.model_2c_bl import Model as Model_bl
# from model.model_2c_att import Model as Model_att
from model.model_2c_board import Model as Model_board
# from model.model_2c_lsa import Model as Model_lsa
# from model.model_2c_idcnn import Model as Model_idcnn
# from model.model_2c_mha import Model as Model_mha
# from model.model_2c_dense import Model as Model_dense
