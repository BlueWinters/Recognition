
import tensorflow as tf
import tensorflow.contrib.slim as slim


class VggNet:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    def forward(self, is_train, reuse, args=None):
        with tf.variable_scope('vggnet', reuse=reuse):
            body = self.x
            body = tf.layers.conv2d(body, 64, (3, 3), (1, 1), padding='same', activation=tf.nn.relu, name='conv2d_1_1')
            body = tf.layers.conv2d(body, 128, (3, 3), (1, 1), padding='same', activation=tf.nn.relu, name='conv2d_1_2')
            body = tf.layers.max_pooling2d(body, (2, 2), (2, 2), padding='valid', name='maxpool_1')

            body = tf.layers.conv2d(body, 256, (3, 3), (1, 1), padding='same', activation=tf.nn.relu, name='conv2d_2_1')
            body = tf.layers.conv2d(body, 256, (3, 3), (1, 1), padding='same', activation=tf.nn.relu, name='conv2d_2_2')
            body = tf.layers.max_pooling2d(body, (2, 2), (2, 2), padding='valid', name='maxpool_2')

            body = tf.layers.conv2d(body, 512, (3, 3), (1, 1), padding='same', activation=tf.nn.relu, name='conv2d_3_1')
            body = tf.layers.conv2d(body, 512, (3, 3), (1, 1), padding='same', activation=tf.nn.relu, name='conv2d_3_2')
            body = tf.layers.max_pooling2d(body, (2, 2), (2, 2), padding='valid', name='maxpool_3')

            body = tf.layers.conv2d(body, 512, (3, 3), (1, 1), padding='same', activation=tf.nn.relu, name='conv2d_4_1')
            body = tf.layers.conv2d(body, 512, (3, 3), (1, 1), padding='same', activation=tf.nn.relu, name='conv2d_4_2')
            body = tf.layers.max_pooling2d(body, (2, 2), (2, 2), padding='valid', name='maxpool_4')

            # flatten
            body = slim.flatten(body, scope='flatten')
            # output probability
            logit = tf.layers.dense(body, 10, name='logits')
            prediction = tf.nn.softmax(logit)

            # loss
            l2_vars = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=self.y))
            weight_decay = 0.0001
            loss = weight_decay * l2_vars + cross_entropy

            return loss, prediction
