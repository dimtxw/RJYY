import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.contrib import cudnn_rnn
import math


class BaseNet(object):
    def __init__(self):
        self.REQUIRED_UNTRAINABLE_VARIABLES = 'required_untrainable_variables'
        self.UPDATE_OPS_COLLECTION = 'update_ops'
        self.BN_DECAY = 0.99
        self.BN_EPSILON = 1e-12
        self.is_training = False

    def relu(self, x):
        if self.is_training:
            return tf.nn.leaky_relu(x)
        else:
            return tf.nn.leaky_relu(x)

    def linear(self, x, output_dims, name="linear", use_bias=True, initializer=None):
        x_shape = x.get_shape()
        input_dims = x_shape[-1]

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            if initializer is None:
                initializer = tf.contrib.layers.xavier_initializer()
            w = tf.get_variable("weights",
                                shape=[input_dims, output_dims],
                                initializer=initializer)

            if len(x_shape) == 2:
                y = x
            else:
                y = tf.reshape(x, [-1, input_dims])

            y = tf.matmul(y, w)

            if use_bias:
                b = tf.get_variable('biases', shape=[output_dims], initializer=tf.zeros_initializer())
                y = tf.nn.bias_add(y, b)

            y = tf.reshape(y, [-1 if s is None else s for s in x_shape.as_list()[:-1]] + [output_dims])

        return y

    def embeddings(self, x, max, channels, use_tanh=False, name="embeddings", initializer=None):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            if initializer is None:
                initializer = tf.contrib.layers.xavier_initializer()
            w = tf.get_variable("weights",
                                shape=[max, channels],
                                initializer=initializer)
            if use_tanh:
                return tf.nn.tanh(tf.nn.embedding_lookup(w, x))
            else:
                return tf.nn.embedding_lookup(w, x)

    def bilstm(self, x, seq_len, lstm_output_dims=None, lstm_layer_count=1, keep_prob=1.0, name="bilstm"):
        x_shape = x.get_shape()
        input_dims = int(x_shape[-1])
        max_seq_len = int(x_shape[-2])
        u = int(input_dims / 2) if lstm_output_dims is None else lstm_output_dims

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            if len(x_shape) >= 4:
                x = tf.reshape(x, [-1, max_seq_len, input_dims])
                seq_len = tf.reshape(seq_len, [-1])

            for i in range(lstm_layer_count):
                with tf.variable_scope("lstm_layer_" + str(i+1), reuse=tf.AUTO_REUSE):
                    cell_fw = cudnn_rnn.CudnnCompatibleLSTMCell(num_units=u)
                    cell_bw = cudnn_rnn.CudnnCompatibleLSTMCell(num_units=u)

                    if keep_prob < 1.0 and self.is_training:
                        cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=keep_prob)
                        cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=keep_prob)

                    outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                 cell_bw,
                                                                 x,
                                                                 sequence_length=seq_len,
                                                                 dtype=tf.float32)

                    x = tf.concat(outputs, axis=-1)

        if len(x_shape) >= 4:
            return tf.reshape(x, [-1 if s is None else s for s in x_shape.as_list()[:-2]] + [max_seq_len, u * 2])
        else:
            return x

    def bigru(self, x, seq_len, lstm_output_dims=None, lstm_layer_count=1, keep_prob=1.0, name="bigru"):
        x_shape = x.get_shape()
        input_dims = int(x_shape[-1])
        max_seq_len = int(x_shape[-2])
        u = int(input_dims / 2) if lstm_output_dims is None else lstm_output_dims

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            if len(x_shape) >= 4:
                x = tf.reshape(x, [-1, max_seq_len, input_dims])
                seq_len = tf.reshape(seq_len, [-1])

            for i in range(lstm_layer_count):
                with tf.variable_scope("lstm_layer_" + str(i+1), reuse=tf.AUTO_REUSE):
                    cell_fw = cudnn_rnn.CudnnCompatibleGRUCell(num_units=u)
                    cell_bw = cudnn_rnn.CudnnCompatibleGRUCell(num_units=u)

                    if keep_prob < 1.0 and self.is_training:
                        cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=keep_prob)
                        cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=keep_prob)

                    outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                 cell_bw,
                                                                 x,
                                                                 sequence_length=seq_len,
                                                                 dtype=tf.float32)

                    x = tf.concat(outputs, axis=-1)

        if len(x_shape) >= 4:
            return tf.reshape(x, [-1 if s is None else s for s in x_shape.as_list()[:-2]] + [max_seq_len, u * 2])
        else:
            return x

    def bilstm_with_c(self, x, seq_len, lstm_output_dims=None, lstm_layer_count=1, keep_prob=1.0, name="bilstm"):
        x_shape = x.get_shape()
        input_dims = int(x_shape[-1])
        max_seq_len = int(x_shape[-2])
        u = int(input_dims / 2) if lstm_output_dims is None else lstm_output_dims

        contexts = []
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            if len(x_shape) >= 4:
                x = tf.reshape(x, [-1, max_seq_len, input_dims])
                seq_len = tf.reshape(seq_len, [-1])

            for i in range(lstm_layer_count):
                with tf.variable_scope("lstm_layer_" + str(i+1), reuse=tf.AUTO_REUSE):
                    cell_fw = cudnn_rnn.CudnnCompatibleLSTMCell(num_units=u)
                    cell_bw = cudnn_rnn.CudnnCompatibleLSTMCell(num_units=u)

                    if keep_prob < 1.0 and self.is_training:
                        cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=keep_prob)
                        cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=keep_prob)

                    outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                     cell_bw,
                                                                     x,
                                                                     sequence_length=seq_len,
                                                                     dtype=tf.float32)
                    contexts.append(tf.concat([state[0].c, state[1].c], axis=-1))

                    x = tf.concat(outputs, axis=-1)

        if len(x_shape) >= 4:
            return \
                tf.reshape(x, [-1 if s is None else s for s in x_shape.as_list()[:-2]] + [max_seq_len, u * 2]), \
                [tf.reshape(c, [-1 if s is None else s for s in x_shape.as_list()[:-2]] + [u * 2]) for c in contexts]
        else:
            return x, contexts

    def context_add(self, context, target, name="context_add"):
        # context与target的特征拼接在一起，计算残差加到context
        # context与target形状可能不一样不能直接拼接

        t_shape = target.get_shape()
        c_shape = context.get_shape()

        if len(t_shape) == 3 and len(c_shape) == 2:
            context = tf.expand_dims(context, 1)

        d_model = int(c_shape[-1])

        with tf.variable_scope(name):
            with tf.variable_scope("l1"):
                with tf.variable_scope("context"):
                    h = self.linear(tf.nn.relu(context), d_model, use_bias=False)
                with tf.variable_scope("target"):
                    h = h + self.linear(target, d_model, use_bias=True)
            with tf.variable_scope("l2"):
                h = tf.nn.relu(h)
                h = self.linear(h, d_model, use_bias=True)

            context = context + h

        return context

    def task_attention(self, target, attention_count,
                       target_len=None,
                       target_attention_window=None,
                       target_special_mask=None,
                       name="task_attention"):

        with tf.variable_scope(name):
            t_shape = target.get_shape()
            target_max_seq_len = int(t_shape[-2])
            if target_len is not None:
                mask_s = tf.expand_dims(
                    (1 - tf.sequence_mask(target_len, target_max_seq_len, dtype=tf.float32)), -2)  # [b, 1, s]
            elif target_attention_window is not None:
                mask_1 = tf.sequence_mask(target_attention_window[0], target_max_seq_len, dtype=tf.float32)
                mask_2 = tf.sequence_mask(target_attention_window[1], target_max_seq_len, dtype=tf.float32)
                mask_s = (1 - tf.abs(mask_2 - mask_1))  # [b, r, s]
            else:
                mask_s = None

            if target_special_mask is not None:
                if mask_s is not None:
                    mask_s = tf.sign(mask_s + target_special_mask)
                else:
                    mask_s = target_special_mask

            if mask_s is not None:
                mask_s = mask_s * 1e12

            query_len = int(mask_s.get_shape()[1])

            a = self.linear(target, attention_count, use_bias=False)  # [b, s, c]
            a = tf.transpose(a, [0, 2, 1])  # [b, c, s]
            a = tf.expand_dims(a, 1)  # [b, 1, c, s]
            if mask_s is not None:
                with tf.variable_scope("mask"):
                    a = a - tf.expand_dims(mask_s, 2)    # [b, r, c, s]

            with tf.variable_scope("softmax"):
                a = tf.nn.softmax(a)

            a = tf.matmul(tf.reshape(a, [-1, query_len * attention_count, target_max_seq_len]), target)  # [b, r*c, n]

            if query_len == 1:
                return tf.reshape(a, [-1, attention_count * int(t_shape[-1])])   # [b, c * n]
            else:
                return tf.reshape(a, [-1, query_len, attention_count * int(t_shape[-1])])  # [b, r, c * n]

    def query_attention(self, query, target,
                        query_len=None,
                        target_len=None,
                        target_attention_window=None,
                        target_special_mask=None,
                        name="query_attention"):
        with tf.variable_scope(name):
            q_shape = query.get_shape()
            t_shape = target.get_shape()
            if len(q_shape) == len(t_shape) - 1:
                query = tf.expand_dims(query, -2)
            d_k = int(target.get_shape()[-1])

            if query_len is not None:
                query_max_seq_len = int(q_shape[-2])
                mask_m = tf.expand_dims(tf.sequence_mask(query_len, query_max_seq_len, dtype=tf.float32), -1)
            else:
                mask_m = None

            if target_len is not None:
                target_max_seq_len = int(t_shape[-2])
                mask_s = tf.expand_dims(
                    (1 - tf.sequence_mask(target_len, target_max_seq_len, dtype=tf.float32)), -2)
            elif target_attention_window is not None:
                target_max_seq_len = int(t_shape[-2])
                mask_1 = tf.sequence_mask(target_attention_window[0], target_max_seq_len, dtype=tf.float32)
                mask_2 = tf.sequence_mask(target_attention_window[1], target_max_seq_len, dtype=tf.float32)
                mask_s = (1 - tf.abs(mask_2 - mask_1))  # [b, r, s]
            else:
                mask_s = None

            if target_special_mask is not None:
                if mask_s is not None:
                    mask_s = tf.sign(mask_s + target_special_mask)
                else:
                    mask_s = target_special_mask

            if mask_s is not None:
                mask_s = mask_s * 1e12

            with tf.variable_scope("Q"):
                q = self.linear(query, d_k, use_bias=False)

            a = tf.matmul(q, target, transpose_b=True) / math.sqrt(d_k)
            if mask_s is not None:
                with tf.name_scope("attention_mask"):
                    a = a - mask_s
            with tf.name_scope("softmax"):
                a = tf.nn.softmax(a)

            a = tf.matmul(a, target)

            if mask_m is not None:
                with tf.name_scope("output_mask"):
                    a = tf.multiply(a, mask_m)

            if len(q_shape) == len(t_shape) - 1:
                return tf.squeeze(a, axis=-2)
            else:
                return a
