import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import tensorflow_probability as tfp
import pickle
import numpy as np

class create_recurrent_unit(object):
    def __init__(self, params):
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

    def unit(self, x, hidden_memory_tm1):
        previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)
        # Input Gate
        print(x.shape, self.Wi.shape)
        print(previous_hidden_state.shape, self.Ui.shape, self.bi.shape)
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

class create_output_unit():
    def __init__(self, params):

        self.Wo = tf.Variable(params[13])
        self.bo = tf.Variable(params[14])
        self.params = []
        self.params.extend([self.Wo, self.bo])

    def unit(self, hidden_memory_tuple):
        hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
        # hidden_state : batch x hidden_dim
        logits = tf.matmul(hidden_state, self.Wo) + self.bo
        # output = tf.nn.softmax(logits)
        return logits  # [batch, vocb_size] logits


batch_size = 5
hidden_dim = 32
sequence_length = 20
start_token =  tf.constant([0] * batch_size, dtype=tf.int32)
vocab_size = 5000# test Monte Carlo
params = pickle.load(open("./save/target_params_py3.pkl", "rb"))
for param in params:
    print(param.shape)
print(params[0][0,:10])

# initial states
h0 = tf.zeros([batch_size, hidden_dim])
h0 = tf.stack([h0, h0])

# generator on initial randomness
gen_o = tensor_array_ops.TensorArray(dtype=tf.float32, size=sequence_length,
                                     dynamic_size=False, infer_shape=True)       # [sequence_len, batch]  probability
gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=sequence_length,
                                     dynamic_size=False, infer_shape=True)       # [sequence_len, batch]  index

# RL process
g_embeddings = tf.Variable(params[0])
g_params = params
# g_params.append(g_embeddings)
g_recurrent_unit_fn = create_recurrent_unit(g_params)
g_output_unit_fn = create_output_unit(g_params)

def _g_recurrence(i, x_t, h_tm1, gen_o, gen_x):
    h_t = g_recurrent_unit_fn.unit(x_t, h_tm1)  # lstm(x_t, h_{t-1}) ->h_t. [2, batch. hidden_size], hidden_memory_tuple
    o_t = g_output_unit_fn.unit(h_t)            # [batch, vocab_size] , logits not prob
    log_prob = tf.math.log(tf.nn.softmax(o_t))  # log-prob  # [batch, vocab_size]

    # Monte Carlo search? 多项分布 Multinomial
    token_search = tfp.distributions.Multinomial(total_count=1, logits=log_prob)  # batch-class distribution
    next_token = tf.cast(tf.argmax(token_search.probs, axis=-1), tf.int32)        # [batch]

    x_tp1 = tf.nn.embedding_lookup(g_embeddings, next_token)  # x_{t+1}, [batch, emb_dim]
    gen_o = gen_o.write(index=i,
                        value=tf.reduce_sum(tf.multiply(tf.one_hot(next_token, vocab_size, 1.0, 0.0),
                                                        tf.nn.softmax(o_t)), 1))  # [batch_size] , prob
    gen_x = gen_x.write(i, next_token)  # indices, batch_size
    return i + 1, x_tp1, h_t, gen_o, gen_x

n, _, _, gen_o, gen_x = control_flow_ops.while_loop(
    cond=lambda i, _1, _2, _3, _4: i < sequence_length,
    body=_g_recurrence,
    loop_vars=(tf.constant(0, dtype=tf.int32),
               tf.nn.embedding_lookup(g_embeddings, start_token), h0, gen_o, gen_x)  # 分别对应 x_t, h_{t-1}, gen_o, gen_x
    )

assert n.numpy() == np.array(sequence_length)

gen_x = gen_x.stack()  # seq_length x batch_size
gen_x = tf.transpose(gen_x, perm=[1, 0])  # batch_size x seq_length
print(gen_x)
gen_o = gen_o.stack()  # sequence x batch_size
gen_o = tf.transpose(gen_o, perm=[1, 0])  # batch_size x seq_length
print(gen_o)


