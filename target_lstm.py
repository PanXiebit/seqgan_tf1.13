import tensorflow as tf
# import tensorflow_probability as tfp
import numpy as np
# tf.enable_eager_execution()

class Create_recurrent_unit(tf.keras.layers.Layer):
    def __init__(self, params):
        super(Create_recurrent_unit, self).__init__()
        # Weights and Bias for input and hidden tensor
        self.Wi = tf.Variable(params[1])
        self.Ui = tf.Variable(params[2])
        self.bi = tf.Variable(params[3])

        self.Wf = tf.Variable(params[4])
        self.Uf = tf.Variable(params[5])
        self.bf = tf.Variable(params[6])

        self.Wog = tf.Variable(params[7])
        self.Uog = tf.Variable(params[8])
        self.bog = tf.Variable(params[9])

        self.Wc = tf.Variable(params[10])
        self.Uc = tf.Variable(params[11])
        self.bc = tf.Variable(params[12])
        self.params = []
        self.params.extend([
            self.Wi, self.Ui, self.bi,
            self.Wf, self.Uf, self.bf,
            self.Wog, self.Uog, self.bog,
            self.Wc, self.Uc, self.bc])

    def call(self, x, hidden_memory_tm1):
        previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)
        # Input Gate
        i = tf.sigmoid(
            tf.matmul(x, self.Wi) + tf.matmul(previous_hidden_state, self.Ui) + self.bi
        )

        # Forget Gate
        f = tf.sigmoid(
            tf.matmul(x, self.Wf) +
            tf.matmul(previous_hidden_state, self.Uf) + self.bf
        )
        # Output Gate
        o = tf.sigmoid(
            tf.matmul(x, self.Wog) +
            tf.matmul(previous_hidden_state, self.Uog) + self.bog
        )
        # New Memory Cell
        c_ = tf.nn.tanh(
            tf.matmul(x, self.Wc) +
            tf.matmul(previous_hidden_state, self.Uc) + self.bc
        )
        # Final Memory cell
        c = f * c_prev + i * c_
        # Current Hidden state
        current_hidden_state = o * tf.nn.tanh(c)
        return tf.stack([current_hidden_state, c])  # [2, batch. hidden_size]

class Create_output_unit(tf.keras.layers.Layer):
    def __init__(self, params):
        super(Create_output_unit, self).__init__()
        self.Wo = tf.Variable(params[13])
        self.bo = tf.Variable(params[14])
        self.params = []
        self.params.extend([self.Wo, self.bo])

    def call(self, hidden_memory_tuple):
        hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
        # hidden_state : batch x hidden_dim
        logits = tf.matmul(hidden_state, self.Wo) + self.bo
        # output = tf.nn.softmax(logits)
        return logits  # [batch, vocb_size] logits

class TARGET_LSTM(tf.keras.layers.Layer):
    def __init__(self, vocab_size, batch_size, emb_dim, hidden_dim, sequence_length, start_token, params):
        super(TARGET_LSTM, self).__init__()
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        self.g_params = []
        self.temperature = 1.0
        self.params = params

        tf.random.set_random_seed(66)

        self.g_embeddings = tf.Variable(self.params[0])
        self.g_recurrent_unit = Create_recurrent_unit(self.params)  # maps h_{t-1} to h_t for generator, one step of LSTM
        self.g_output_unit = Create_output_unit(self.params)        # maps h_t to o_t (output token logits), logits

    def generate(self):
        self.h0 = tf.zeros([self.batch_size, self.hidden_dim])
        self.h0 = tf.stack([self.h0, self.h0])

        # generator on initial randomness
        gen_o = tf.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)
        gen_x = tf.TensorArray(dtype=tf.int32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)

        # RL process
        def _g_recurrence(i, x_t, h_tm1, gen_o, gen_x):
            h_t = self.g_recurrent_unit(x_t,h_tm1)  # lstm(x_t, h_{t-1}) ->h_t. [batch. hidden_size * 2], hidden_memory_tuple
            o_t = self.g_output_unit(h_t)  # [batch, vocab_size] , logits not prob
            log_prob = tf.math.log(tf.nn.softmax(o_t))  # log-prob  # [batch, vocab_size]

            # Monte Carlo search? 多项分布 Multinomial
            # self.token_search = tfp.distributions.Multinomial(total_count=1, logits=log_prob)
            # next_token = tf.cast(tf.argmax(self.token_search.probs, axis=-1), tf.int32)  # [batch]
            next_token = tf.cast(tf.reshape(tf.multinomial(logits=log_prob, num_samples=1),
                                            [self.batch_size]), tf.int32)

            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings,next_token)  # x_{t+1}, [batch, emb_dim] 作为下一个 time_step 的输入
            gen_o = gen_o.write(index=i,
                                value=tf.reduce_sum(tf.multiply(tf.one_hot(next_token, self.vocab_size, 1.0, 0.0),
                                                                tf.nn.softmax(o_t)), 1))  # [batch_size] , prob
            gen_x = gen_x.write(i, next_token)  # indices, batch_size
            return i + 1, x_tp1, h_t, gen_o, gen_x

        n, _, _, self.gen_o, self.gen_x = tf.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
            body=_g_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token),
                       self.h0, gen_o, gen_x)  # # 分别对应 time_step, x_t, h_{t-1}, gen_o, gen_x
        )
        # assert n.numpy() == np.array(self.sequence_length)

        self.gen_x = self.gen_x.stack()  # [seq_length, batch_size]
        self.gen_x = tf.transpose(self.gen_x, perm=[1, 0])  # [batch_size, seq_length]
        return self.gen_x

    def pretrain(self, input_x):
        # supervised pretraining for generator
        # processed for batch
        with tf.device("/cpu:0"):
            self.processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, input_x),
                                            perm=[1, 0, 2])  # seq_length x batch_size x emb_dim
        g_predictions = tf.TensorArray(
            dtype=tf.float32, size=self.sequence_length,
            dynamic_size=False, infer_shape=True)

        ta_emb_x = tf.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        ta_emb_x = ta_emb_x.unstack(self.processed_x)

        def _pretrain_recurrence(i, x_t, h_tm1, g_predictions):
            h_t = self.g_recurrent_unit(x_t, h_tm1)
            o_t = self.g_output_unit(h_t)
            g_predictions = g_predictions.write(i, tf.nn.softmax(o_t))  # [batch, vocab_size] 生成得到的每一个 time_step 的 probability.
            x_tp1 = ta_emb_x.read(i)                                    # 有监督和无监督的区别，这里 x_{t+1} 是直接读取的 ta_emb_x
            return i + 1, x_tp1, h_t, g_predictions

        _, _, _, self.g_predictions = tf.while_loop(
            cond=lambda i, _1, _2, _3: i < self.sequence_length,
            body=_pretrain_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token),
                       self.h0, g_predictions))   # 分别对应 time_step, x_t, h_{t-1}, g_predictions

        self.g_predictions = tf.transpose(
            self.g_predictions.stack(), perm=[1, 0, 2])  # [batch_size, seq_length, vocab_size]

        # pretraining loss
        self.real_target = tf.one_hot(tf.cast(tf.reshape(input_x, [-1]), tf.int32), self.vocab_size, 1.0, 0.0)
        self.pretrain_loss = -tf.reduce_sum(self.real_target * tf.math.log(tf.reshape(self.g_predictions, [-1, self.vocab_size])))
        self.pretrain_loss = tf.reduce_mean(self.pretrain_loss, axis=None)

        self.out_loss = tf.reduce_sum(
            tf.reshape(
                -tf.reduce_sum(self.real_target * tf.math.log(tf.reshape(self.g_predictions, [-1, self.vocab_size])), 1),
                shape=[-1, self.sequence_length]), 1)  # batch_size
        return self.pretrain_loss

if __name__ == "__main__":
    import pickle
    # tf.enable_eager_execution()
    tmp_params = pickle.load(open("./save/target_params_py3.pkl", "rb"))
    tmp_target_lstm = TARGET_LSTM(vocab_size=5000,
                              batch_size=5,
                              emb_dim=64,
                              hidden_dim=32,
                              sequence_length=5,
                              start_token=0,
                              params=tmp_params)
    # tmp_x = tf.constant([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]], dtype=tf.int32)
    # tmp_target_lstm(tmp_x)
    tmp_example = tmp_target_lstm.generate()
    print(tmp_example)