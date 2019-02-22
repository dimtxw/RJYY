import tensorflow as tf
from model.model_net import ModelNet
from model.model_2c_base import Model as BaseModel
import math


class Net(ModelNet):
    def __init__(self, xci, xtc, xsl, rel, rsl,
                 vocabulary_size=25000,
                 char_embedding_size=128,
                 tag_class_size=16,
                 tag_class_embedding_size=64,
                 is_training=False):
        super(Net, self).__init__(is_training)

        rel_shape = rel.get_shape()
        rel_count = int(rel_shape[1])
        x_shape = xci.get_shape()
        max_seq_len = int(x_shape[1])

        with tf.name_scope("input_encoding"):
            with tf.variable_scope("token_encoding"):
                xce = self.embeddings(xci, vocabulary_size, char_embedding_size,
                                      initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))
                if self.is_training:
                    xce = tf.nn.dropout(xce, 0.5)

            with tf.variable_scope("tag_class_encoding"):
                xte = self.embeddings(xtc, tag_class_size, tag_class_embedding_size,
                                      initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))
                if self.is_training:
                    xte = tf.nn.dropout(xte, 0.5)

            embedding_size = char_embedding_size + tag_class_embedding_size
            position_embeddings = tf.constant([
                [
                    math.sin(p / math.pow(10000, i / embedding_size)) if i % 2 == 0
                    else math.cos(p / math.pow(10000, (i - 1) / embedding_size))
                    for i in range(embedding_size)
                ] for p in range(1, max_seq_len + 1)
            ], dtype=tf.float32, shape=[max_seq_len, embedding_size])

        xem = tf.concat([xce, xte], axis=-1) + position_embeddings

        with tf.variable_scope("main"):
            xem, contexts = self.bilstm_with_c(xem, xsl,
                                               keep_prob=0.5)
            self.lstm_out = xem
            xem = xem + position_embeddings

            # rel形状: (None, rel_count, 2)
            rel_attention_window = (tf.maximum(tf.reduce_min(rel, axis=2)-11, 0),
                                    tf.minimum(tf.reduce_max(rel, axis=2)+12, max_seq_len))

            # rel转换为: (None, rel_count * 2, max_seq_len) 在最后一维是独热的
            rel = tf.one_hot(tf.reshape(rel, [-1, rel_count * 2]), max_seq_len, dtype=tf.float32)
            # 要获得的形状： (None, rel_count * 2, feature_dims)
            # 与xem(None, max_seq_len, feature_dims) 矩阵相乘得到 (None, rel_count * 2, feature_dims)
            kp = tf.matmul(rel, xem)
            feature_dims = int(kp.get_shape()[-1])
            kp = tf.reshape(kp, (-1, rel_count, 2 * feature_dims))

            c = contexts[-1]
            rel_context = self.context_add(c, kp, name="add_entities")
            self.rel_context1 = rel_context

            entity_mask = tf.cast(tf.sign(xtc), dtype=tf.float32)
            qa = self.query_attention(rel_context, xem,
                                      target_attention_window=rel_attention_window,
                                      target_special_mask=tf.expand_dims(entity_mask, -2),
                                      name="context_attention_1")
            rel_context = self.context_add(rel_context, qa, name="add_attention")
            self.rel_context2 = rel_context

            # 再与kp拼接在一起经过前馈网络得到输出
            with tf.variable_scope("out"):
                with tf.variable_scope("l1"):
                    out = self.linear(tf.concat([self.relu(rel_context), kp], axis=-1), 128)
                with tf.variable_scope("l2"):
                    out = self.relu(out)
                    out = self.linear(out, 1)

        self.out = out


class NetS6(ModelNet):
    # lstm输入输出都加PE，不拼接lstm out

    def __init__(self, xci, xtc, xsl, rel, rsl,
                 vocabulary_size=25000,
                 char_embedding_size=128,
                 tag_class_size=16,
                 tag_class_embedding_size=64,
                 is_training=False):
        super(NetS6, self).__init__(is_training)

        rel_shape = rel.get_shape()
        rel_count = int(rel_shape[1])
        x_shape = xci.get_shape()
        max_seq_len = int(x_shape[1])

        with tf.name_scope("input_encoding"):
            with tf.variable_scope("token_encoding"):
                xce = self.embeddings(xci, vocabulary_size, char_embedding_size,
                                      initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))
                if self.is_training:
                    xce = tf.nn.dropout(xce, 0.5)

            with tf.variable_scope("tag_class_encoding"):
                xte = self.embeddings(xtc, tag_class_size, tag_class_embedding_size,
                                      initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))
                if self.is_training:
                    xte = tf.nn.dropout(xte, 0.5)

            embedding_size = char_embedding_size + tag_class_embedding_size
            position_embeddings = tf.constant([
                [
                    math.sin(p / math.pow(10000, i / embedding_size)) if i % 2 == 0
                    else math.cos(p / math.pow(10000, (i - 1) / embedding_size))
                    for i in range(embedding_size)
                ] for p in range(1, max_seq_len + 1)
            ], dtype=tf.float32, shape=[max_seq_len, embedding_size])

        xem = tf.concat([xce, xte], axis=-1) + position_embeddings

        with tf.variable_scope("main"):
            xem, contexts = self.bilstm_with_c(xem, xsl,
                                               keep_prob=0.5)
            self.lstm_out = xem
            xem = xem + position_embeddings

            # rel形状: (None, rel_count, 2)
            rel_attention_window = (tf.maximum(tf.reduce_min(rel, axis=2)-11, 0),
                                    tf.minimum(tf.reduce_max(rel, axis=2)+12, max_seq_len))

            # rel转换为: (None, rel_count * 2, max_seq_len) 在最后一维是独热的
            rel = tf.one_hot(tf.reshape(rel, [-1, rel_count * 2]), max_seq_len, dtype=tf.float32)
            # 要获得的形状： (None, rel_count * 2, feature_dims)
            # 与xem(None, max_seq_len, feature_dims) 矩阵相乘得到 (None, rel_count * 2, feature_dims)
            kp = tf.matmul(rel, xem)
            feature_dims = int(kp.get_shape()[-1])
            kp = tf.reshape(kp, (-1, rel_count, 2 * feature_dims))

            c = contexts[-1]
            rel_context = self.context_add(c, kp, name="add_entities")
            self.rel_context1 = rel_context

            entity_mask = tf.cast(tf.sign(xtc), dtype=tf.float32)
            qa = self.query_attention(rel_context, xem,
                                      target_attention_window=rel_attention_window,
                                      target_special_mask=tf.expand_dims(entity_mask, -2),
                                      name="context_attention_1")
            rel_context = self.context_add(rel_context, qa, name="add_attention")
            self.rel_context2 = rel_context

            # 再与kp拼接在一起经过前馈网络得到输出
            with tf.variable_scope("out"):
                with tf.variable_scope("l1"):
                    out = self.linear(self.relu(rel_context), embedding_size)
                with tf.variable_scope("l2"):
                    out = self.relu(out)
                    out = self.linear(out, 1)

        self.out = out


class Model(BaseModel):
    def build_train_net(self, xci, xtc, xsl, rel, rsl):
        if self.style == "6":
            return NetS6(xci, xtc, xsl, rel, rsl,
                         vocabulary_size=self.vocabulary_size,
                         char_embedding_size=self.char_embedding_size,
                         tag_class_size=self.tag_class_size,
                         tag_class_embedding_size=self.tag_class_embedding_size,
                         is_training=True)
        else:
            return Net(xci, xtc, xsl, rel, rsl,
                       vocabulary_size=self.vocabulary_size,
                       char_embedding_size=self.char_embedding_size,
                       tag_class_size=self.tag_class_size,
                       tag_class_embedding_size=self.tag_class_embedding_size,
                       is_training=True)

    def build_test_net(self, xci, xtc, xsl, rel, rsl):
        if self.style == "6":
            return NetS6(xci, xtc, xsl, rel, rsl,
                         vocabulary_size=self.vocabulary_size,
                         char_embedding_size=self.char_embedding_size,
                         tag_class_size=self.tag_class_size,
                         tag_class_embedding_size=self.tag_class_embedding_size,
                         is_training=False)
        else:
            return Net(xci, xtc, xsl, rel, rsl,
                       vocabulary_size=self.vocabulary_size,
                       char_embedding_size=self.char_embedding_size,
                       tag_class_size=self.tag_class_size,
                       tag_class_embedding_size=self.tag_class_embedding_size,
                       is_training=False)

    def __init__(self,
                 max_train_seq_len=1500,
                 max_test_seq_len=1500,
                 max_train_rel_count=1500,
                 max_test_rel_count=1500,
                 vocabulary_size=3500,
                 char_embedding_size=112,
                 tag_class_size=20,
                 tag_class_embedding_size=16,
                 style=None,
                 ensemble_model=None,
                 graph=None,
                 name="board_model"):
        self.vocabulary_size = vocabulary_size
        self.char_embedding_size = char_embedding_size
        self.tag_class_size = tag_class_size
        self.tag_class_embedding_size = tag_class_embedding_size
        self.style = style

        super(Model, self).__init__(max_train_seq_len=max_train_seq_len,
                                    max_test_seq_len=max_test_seq_len,
                                    max_train_rel_count=max_train_rel_count,
                                    max_test_rel_count=max_test_rel_count,
                                    ensemble_model=ensemble_model,
                                    graph=graph,
                                    name=name)
