import tensorflow as tf


class EnsembleModel(object):
    UPDATE_OPS_COLLECTION = "update_ops"

    def __init__(self,
                 max_train_seq_len=1500,
                 max_test_seq_len=1500,
                 max_train_rel_count=1500,
                 max_test_rel_count=1500,
                 graph=None):
        if graph is None:
            graph = tf.Graph()
        self.graph = graph

        with graph.as_default():
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

            step = tf.get_variable("step", shape=(), initializer=tf.zeros_initializer, dtype=tf.int32)
            self.model_step = step
            tf.add_to_collection(self.UPDATE_OPS_COLLECTION, tf.assign_add(step, 1))

        self.children_models = []

    def set_model_step(self, value=0.0):
        return [m.set_model_step(value) for m in self.children_models]

    def get_train_update_ops(self):
        with self.graph.as_default():
            return tf.get_collection(self.UPDATE_OPS_COLLECTION)