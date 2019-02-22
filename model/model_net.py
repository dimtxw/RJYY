import tensorflow as tf
import numpy as np
from model.base_net import BaseNet


class ModelNet(BaseNet):
    def __init__(self, is_training):
        super(ModelNet, self).__init__()
        self.is_training = is_training
        self.watching_1 = []
        self.watching_g = []
        self.short_hard_outs = []

    def add_watching(self, x, name=None):
        if name is not None:
            self.watching_1.append([
                tf.reduce_max(x, name=name + "_max"),
                tf.reduce_mean(x, name=name + "_mean"),
                tf.reduce_min(x, name=name + "_min")
            ])
        else:
            self.watching_1.append([
                tf.reduce_max(x),
                tf.reduce_mean(x),
                tf.reduce_min(x)
            ])

    def add_gradient_watching(self, x, name):
        x_shape = x.get_shape()
        v_shape = []
        for i in range(len(x_shape)):
            if x_shape[i].value is None:
                v_shape.append(1)
            else:
                v_shape.append(int(x_shape[i]))

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            g = tf.Variable(np.zeros(v_shape, dtype=np.float32),
                            name="gradient_watchings",
                            dtype=tf.float32,
                            trainable=False)
            self.watching_g.append(g)

        x = x + g
        return x