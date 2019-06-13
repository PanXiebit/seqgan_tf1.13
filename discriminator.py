import tensorflow as tf
import numpy as np

# An alternative to tf.nn.rnn_cell._linear function, which has been removed in Tensorfow 1.0.1
# The highway layer is borrowed from https://github.com/mkroutikov/tf-lstm-char-cnn

class Highway(tf.keras.layers.Layer):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    g = relu(Wy+b)
    z = t * g + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """
    def __init__(self, input_size, num_layers):
        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.g_linears = []
        self.t_linears = []
        for i in range(num_layers):
            self.g_linears.append(tf.keras.layers.Dense(input_size, use_bias=True))
            self.t_linears.append(tf.keras.layers.Dense(input_size, use_bias=True))

    def call(self, input_):
        output = input_
        for idx in range(self.num_layers):
            g = tf.nn.relu(self.g_linears[idx](output))
            t = tf.nn.sigmoid(self.t_linears[idx](output))
            output = t * g + (1.0 - t) * output
        return output


class Discriminator(tf.keras.layers.Layer):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, seq_len, vocab_size, embed_size, filter_sizes, num_filters, num_classes,
                 l2_reg_lambda=0.0, dropout_keep_prob=0.1, d_lr_rate=1e-4, grad_clip=5):
        super(Discriminator, self).__init__()
        self.filter_sizes = filter_sizes  # [3,4,5]
        self.num_filters = num_filters    # [128, 128, 128]
        self.embed_size = embed_size
        self.learning_rate = d_lr_rate
        self.grad_clip = grad_clip

        # embedding layer
        with tf.variable_scope("discriminator"):
            self.embedding = tf.random_uniform_initializer(maxval=0.8, minval=-0.8)(
                shape=(vocab_size, embed_size), dtype=tf.float32)

            # convolution layer
            self.Convs = []
            self.Pools = []
            for filter_size, num_filter in zip(self.filter_sizes, self.num_filters):
                conv = tf.keras.layers.Conv2D(filters=num_filter, kernel_size=[filter_size, embed_size],
                                              strides=(1,1), padding="valid", use_bias=True)
                self.Convs.append(conv)
                pool = tf.keras.layers.MaxPool2D(pool_size=(seq_len - filter_size + 1, 1),
                                                 strides=(1,1),
                                                 padding="valid")
                self.Pools.append(pool)

            # highway and dropout
            self.num_filters_total = sum(self.num_filters)
            self.Highway = Highway(self.num_filters_total, num_layers=1)
            self.Dropout = tf.keras.layers.Dropout(rate=dropout_keep_prob)

            # prediction and score
            self.pred_linear = tf.keras.layers.Dense(num_classes, use_bias=True)

        # loss
        self.loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction="none")
        # Keeping track of l2 regularization loss (optional)
        self.l2_loss = tf.constant(0.0)
        self.l2_reg_lambda = l2_reg_lambda
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

    def get_logits(self, input_x):
        embed_chars = tf.nn.embedding_lookup(self.embedding, input_x)  # [batch, x_seq_len, embed_size]
        embed_chars_expanded = tf.expand_dims(embed_chars, axis=-1)    # [batch, x_seq_len, embed_size, 1]

        pooled_outs = []
        for i in range(len(self.Convs)):
            conv_out = self.Convs[i](embed_chars_expanded)    # [batch, x_seq_len - filter_size + 1, 1, num_filter]
            # applay nonlinearity
            h = tf.nn.relu(conv_out)
            pooled = self.Pools[i](h)                         # [batch, 1, 1, num_filters]
            pooled_outs.append(pooled)

        # combine all the pooled features
        self.h_pool = tf.concat(pooled_outs, axis=3)               # [batch, 1, 1, num_filters_total]
        #self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])
        self.h_pool_flat = tf.squeeze(self.h_pool)                 # [batch, num_filters_total]
        assert self.h_pool_flat.shape[-1] == self.num_filters_total
        # add highway
        self.h_highway = self.Highway(self.h_pool_flat)            # [batch, num_filters_total]
        # add dropout
        self.h_drop = self.Dropout(self.h_highway)                 # [batch, num_filters_total]

        # Final (unnormalized) scores and predictions
        self.scores = self.pred_linear(self.h_drop)                 # [batch, num_calsses]
        self.ypred_for_auc = tf.nn.softmax(self.scores)

        self.predictions = tf.cast(tf.argmax(self.scores, axis=1), tf.int32)  # [batch,]
        return self.ypred_for_auc, self.predictions

    def comput_loss(self, input_x, input_y):
        ypred_for_auc, _ = self.get_logits(input_x)
        losses = tf.reduce_mean(self.loss_obj(y_pred=ypred_for_auc, y_true=input_y))  # [batch] -> 1
        tf.identity(losses, name="cross_entropy")
        for weights in self.pred_linear.trainable_weights:
            self.l2_loss += tf.nn.l2_loss(weights)
        self.loss = losses + self.l2_reg_lambda * self.l2_loss
        tf.identity(self.loss, name="total_loss")
        return self.loss

    def train_op(self, input_x, input_y):
        with tf.GradientTape() as tape:
            d_loss = self.comput_loss(input_x, input_y)
            self.pretrain_grad, _ = tf.clip_by_global_norm(
                tape.gradient(d_loss, d_loss.trainable_variables), self.grad_clip)
            self.optimizer.apply_gradients(zip(self.pretrain_grad, self.d_params))
        return d_loss

if __name__ == "__main__":
    # test discriminator
    tf.enable_eager_execution()
    tmp_discriminator = Discriminator(seq_len=5, vocab_size=1000, embed_size=64, filter_sizes=[3,4,5],
                                      num_filters=[128, 128, 128], num_classes=2)
    tmp_input_x = tf.constant(value=[[1,2,3,4,5],[2,4,6,8,0]], dtype=tf.int32)
    tmp_input_y = tf.constant(value=[[0,1], [0,1]], dtype=tf.int32)
    ypred_for_auc, _ = tmp_discriminator.get_logits(tmp_input_x)
    print(ypred_for_auc.shape, ypred_for_auc)
    tmp_loss = tmp_discriminator.comput_loss(tmp_input_x, tmp_input_y)
    print(tmp_loss)
    tmp_discriminator.train_op(tmp_input_x, tmp_input_y)



