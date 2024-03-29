import tensorflow as tf

class Generator(tf.keras.Model):
    def __init__(self, vocab_size, batch_size, emb_dim, hidden_dim,
                 sequence_length, start_token,
                 learning_rate=0.01, reward_gamma=0.95):
        super(Generator, self).__init__()
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.reward_gamma = reward_gamma
        self.g_params = []
        self.grad_clip = 5.0

        # initilizate state
        self.h0 = tf.zeros([self.batch_size, self.hidden_dim])
        self.h0 = tf.stack([self.h0, self.h0])

        # define variables
        self.g_embeddings = tf.Variable(self.init_matrix([self.vocab_size, self.emb_dim]))
        self.g_params.append(self.g_embeddings)
        self.g_recurrent_unit = self.create_recurrent_unit(self.g_params)  # maps h_{t-1} to h_t for generator
        self.g_output_unit = self.create_output_unit(self.g_params)  # maps h_t to o_t (output token logits)

    def _unsuper_generate(self):
        """ unsupervised generate. using in rollout policy.
        :return: 生成得到的 token index
        """
        """
        :param input_x:  [batch, seq_len]
        :param rewards:  [batch, seq_len]
        :return:
        """
        gen_o = tf.TensorArray(dtype=tf.float32, size=self.sequence_length,
                               dynamic_size=False, infer_shape=True)
        gen_x = tf.TensorArray(dtype=tf.int32, size=self.sequence_length,
                               dynamic_size=False, infer_shape=True)

        def _g_recurrence(i, x_t, h_tm1, gen_o, gen_x):
            h_t = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
            o_t = self.g_output_unit(h_t)  # [batch, vocab] , logits not prob
            log_prob = tf.log(tf.nn.softmax(o_t))
            #tf.logging.info("unsupervised generated log_prob:{}".format(log_prob[0]))
            next_token = tf.cast(tf.reshape(tf.multinomial(logits=log_prob, num_samples=1),
                                            [self.batch_size]), tf.int32)
            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)  # [batch, emb_dim]
            gen_o = gen_o.write(i, tf.reduce_sum(tf.multiply(tf.one_hot(next_token, self.vocab_size, 1.0, 0.0),
                                                             tf.nn.softmax(o_t)), 1))  # [batch_size] , prob
            gen_x = gen_x.write(i, next_token)  # indices, batch_size
            return i + 1, x_tp1, h_t, gen_o, gen_x

        _, _, _, self.gen_o, self.gen_x = tf.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
            body=_g_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token),
                       self.h0, gen_o, gen_x))

        self.gen_x = self.gen_x.stack()  # [seq_length, batch_size]
        self.gen_x = tf.transpose(self.gen_x, perm=[1, 0])  # [batch_size, seq_length]
        return self.gen_x

    def _super_generate(self, input_x):
        """ supervised generate.

        :param input_x:
        :return: 生成得到的是 probability [batch * seq_len, vocab_size]
        """
        with tf.device("/cpu:0"):
            self.processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, input_x),
                                            perm=[1, 0, 2])  # [seq_len, batch_size, emb_dim]
        # supervised pretraining for generator
        g_predictions = tf.TensorArray(
            dtype=tf.float32, size=self.sequence_length,
            dynamic_size=False, infer_shape=True)

        ta_emb_x = tf.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        ta_emb_x = ta_emb_x.unstack(self.processed_x)

        def _pretrain_recurrence(i, x_t, h_tm1, g_predictions):
            h_t = self.g_recurrent_unit(x_t, h_tm1)
            o_t = self.g_output_unit(h_t)
            g_predictions = g_predictions.write(i, tf.nn.softmax(o_t))  # [batch, vocab_size]
            x_tp1 = ta_emb_x.read(i)                                    # supervised learning, teaching forcing.
            return i + 1, x_tp1, h_t, g_predictions

        _, _, _, self.g_predictions = tf.while_loop(
            cond=lambda i, _1, _2, _3: i < self.sequence_length,
            body=_pretrain_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token),
                       self.h0, g_predictions))
        self.g_predictions = tf.transpose(self.g_predictions.stack(),
                                          perm=[1, 0, 2])  # [batch_size, seq_length, vocab_size]
        self.g_predictions = tf.clip_by_value(
            tf.reshape(self.g_predictions, [-1, self.vocab_size]), 1e-20, 1.0)  # [batch_size*seq_length, vocab_size]
        return self.g_predictions       # [batch_size*seq_length, vocab_size]

    def _get_pretrain_loss(self, input_x):
        """
        :param input_x:  # [batch, seq_len]
        :return:
        """
        self.g_predictions = self._super_generate(input_x)
        real_target = tf.one_hot(
            indices=tf.to_int32(tf.reshape(input_x, [-1])),
            depth=self.vocab_size, on_value=1.0, off_value=0.0)  # [batch_size * seq_length, vocab_size]
        self.pretrain_loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.g_predictions,
            labels=real_target)                                  # [batch_size * seq_length]
        self.pretrain_loss = tf.reduce_mean(self.pretrain_loss)  # scalar
        return self.pretrain_loss  # [batch * seq_length]

    def _get_generate_loss(self, input_x, rewards):
        """

        :param input_x: [batch, seq_len]
        :param rewards: [batch, seq_len]
        :return:
        """
        self.g_predictions = self._super_generate(input_x)
        real_target = tf.one_hot(
            tf.to_int32(tf.reshape(input_x, [-1])),
            depth=self.vocab_size, on_value=1.0, off_value=0.0)  # [batch_size * seq_length, vocab_size]
        self.pretrain_loss = tf.nn.softmax_cross_entropy_with_logits(labels=real_target,
                                                                     logits=self.g_predictions)  # [batch * seq_length]
        self.g_loss = tf.reduce_mean(self.pretrain_loss * tf.reshape(rewards, [-1]))  # scalar
        return self.g_loss


    # def pretrain_step(self, input_x):
    #     """ teaching forcing generate
    #     :param input_x: [batch, seq_len]
    #     :return:
    #     """
    #     pretrain_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    #     with tf.GradientTape() as tape:
    #         # https://stackoverflow.com/questions/50244706/trying-to-call-tape-gradient-on-a-non-persistent-tape-while
    #         # -it-is-still-active
    #         # 使用 tg.GradientTape 时， loss 的计算必须在里面
    #         pretrain_loss = tf.reduce_mean(self.super_generate(input_x))    # scalar
    #         gradients = tape.gradient(pretrain_loss, self.g_params)
    #         self.pretrain_grad, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
    #         pretrain_optimizer.apply_gradients(zip(self.pretrain_grad, self.g_params))
    #     return pretrain_loss
    #
    # def update_step_by_reward(self, input_x, rewards):
    #     """
    #     :param input_x: [batch, seq_len]
    #     :param rewards: [batch, seq_len]
    #     :return:
    #     """
    #     # Unsupervised training
    #     g_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    #     with tf.GradientTape() as tape:
    #         g_loss = tf.reduce_sum(self.super_generate(input_x) * tf.reshape(rewards, [-1]))
    #         self.pretrain_grad, _ = tf.clip_by_global_norm(
    #             tape.gradient(g_loss, self.g_params), self.grad_clip)
    #         g_optimizer.apply_gradients(zip(self.pretrain_grad, self.g_params))
    #     return g_loss

    def init_matrix(self, shape):
        return tf.random.normal(shape, stddev=0.1)

    def create_recurrent_unit(self, params):
        # Weights and Bias for input and hidden tensor
        # input/update gate
        self.Wi = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Ui = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bi = tf.Variable(self.init_matrix([self.hidden_dim]))

        # forget gate
        self.Wf = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uf = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bf = tf.Variable(self.init_matrix([self.hidden_dim]))

        # output gate
        self.Wog = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uog = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bog = tf.Variable(self.init_matrix([self.hidden_dim]))

        # new memory cell
        self.Wc = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uc = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bc = tf.Variable(self.init_matrix([self.hidden_dim]))
        params.extend([
            self.Wi, self.Ui, self.bi,
            self.Wf, self.Uf, self.bf,
            self.Wog, self.Uog, self.bog,
            self.Wc, self.Uc, self.bc])

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)  # h_{t-1}, c_{t-1}
            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi  #[batch, hidden_size]
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

            return tf.stack([current_hidden_state, c])   # [batch, hidden_state * 2]
        return unit  # [batch, hidden_state * 2]

    def create_output_unit(self, params):
        self.Wo = tf.Variable(self.init_matrix([self.hidden_dim, self.vocab_size]))
        self.bo = tf.Variable(self.init_matrix([self.vocab_size]))
        params.extend([self.Wo, self.bo])

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit


if __name__ == "__main__":
    # tf.enable_eager_execution()
    tmp_generator = Generator(vocab_size=1000, batch_size=5, emb_dim=64, hidden_dim=64,
                          sequence_length=20, start_token=0,
                          learning_rate=0.01, reward_gamma=0.95)
    # for module in tmp_generator.trainable_variables:
    #     print(module)
    tmp_example = tmp_generator._unsuper_generate()  # [5, 20]
    print(tmp_example.shape)
    # # pretrain generator using tmp_example
    # tmp_example2 = tmp_generator._super_generate(tmp_example)
    # print(tmp_example2.shape)
    # print(tmp_generator.trainable_variables)
    g_optimizer = tf.train.AdamOptimizer(learning_rate=tmp_generator.learning_rate)
    def gen_train_step(x_batch):
        with tf.GradientTape() as tape:
            # https://stackoverflow.com/questions/50244706/trying-to-call-tape-gradient-on-a-non-persistent-tape-
            # while-it-is-still-active
            # 使用 tg.GradientTape 时， loss 的计算必须在里面
            pretrain_g_loss = tmp_generator._get_pretrain_loss(x_batch)
            print(pretrain_g_loss)
            print(tmp_generator.trainable_variables)
            g_gradients, _ = tf.clip_by_global_norm(
                tape.gradient(pretrain_g_loss, tmp_generator.trainable_variables), clip_norm=5.0)
            g_optimizer.apply_gradients(zip(g_gradients, tmp_generator.trainable_variables))
        return pretrain_g_loss

    for i in range(10):
        gen_train_step(tmp_example)