import tensorflow as tf
# tf.enable_eager_execution()


class test_model(tf.keras.Model):
    def __init__(self, num_classes):
        super(test_model, self).__init__()
        self.num_classes = num_classes
        self.linear = tf.keras.layers.Dense(num_classes)
        self.linear2 = tf.keras.layers.Dense(num_classes)
        self.loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    def call(self, inputs, labels, training=False):
        """

        :param inputs: [batch, 20]
        :param labels: [batch]
        :return:
        """
        logits = self.linear2(self.linear(inputs))
        real_labels = tf.one_hot(indices=labels, depth=self.num_classes, on_value=1, off_value=0)
        loss = self.loss(y_true=real_labels, y_pred=logits)
        return loss, logits

tmp_inputs = tf.random.normal(shape=[5, 20], dtype=tf.float32)
tmp_label = tf.constant([0, 4, 3, 2, 3], dtype=tf.int32)
tmp_model = test_model(5)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

# @tf.function
def train_step():
    with tf.GradientTape() as tape:
        tmp_loss, predictions = tmp_model(tmp_inputs, tmp_label)
        # gradients = tape.gradient(tmp_loss, tmp_model.variables)
        # print(gradients)
        print(tmp_loss)
        print("variables:{}".format(tmp_model.linear2.trainable_variables))
        clip_grads, _ = tf.clip_by_global_norm(
            tape.gradient(tmp_loss, tmp_model.linear2.trainable_variables), clip_norm=5)
        # print(clip_grads)
        optimizer.apply_gradients(zip(clip_grads, tmp_model.linear2.trainable_variables))

if __name__ == "__main__":
    NUM_EPOCH = 10
    for i in range(NUM_EPOCH):
        train_step()
