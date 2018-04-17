
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time as time
import tools.cifar10 as cifar10
import tools.helper as helper




class VggNet:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        self.ratio = 8

    def se_conv2d(self, input, out_chl, kernel, stride, padding, activation, name, is_train):
        with tf.variable_scope(name):
            ratio = self.ratio
            assert out_chl % ratio == 0
            body = tf.layers.conv2d(input, out_chl, kernel, stride, padding=padding, activation=activation, name='conv2d')
            body = tf.layers.batch_normalization(body, momentum=0.9, training=is_train, name='bn')
            squeeze = tf.reduce_mean(body, axis=(1, 2), name='squeeze')
            excitation = tf.layers.dense(squeeze, units=int(out_chl/ratio), activation=activation, name='dense1')
            excitation = tf.layers.dense(excitation, units=int(out_chl), activation=tf.nn.sigmoid, name='dense2')
            excitation = tf.reshape(excitation, shape=(-1, 1, 1, out_chl))
            output = body * excitation
            return output
            # return body

    def forward(self, is_train, reuse, args=None):
        with tf.variable_scope('vggnet', reuse=reuse):
            body = self.x
            body = self.se_conv2d(body, 64, (3, 3), (1, 1), 'same', tf.nn.relu, 'conv2d_1_1', is_train)
            body = self.se_conv2d(body, 128, (3, 3), (1, 1), 'same', tf.nn.relu, 'conv2d_1_2', is_train)
            body = tf.layers.max_pooling2d(body, (2, 2), (2, 2), padding='valid', name='maxpool_1')

            body = self.se_conv2d(body, 256, (3, 3), (1, 1), 'same', tf.nn.relu, 'conv2d_2_1', is_train)
            body = self.se_conv2d(body, 256, (3, 3), (1, 1), 'same', tf.nn.relu, 'conv2d_2_2', is_train)
            body = tf.layers.max_pooling2d(body, (2, 2), (2, 2), padding='valid', name='maxpool_2')

            body = self.se_conv2d(body, 512, (3, 3), (1, 1), 'same', tf.nn.relu, 'conv2d_3_1', is_train)
            body = self.se_conv2d(body, 512, (3, 3), (1, 1), 'same', tf.nn.relu, 'conv2d_3_2', is_train)
            body = tf.layers.max_pooling2d(body, (2, 2), (2, 2), padding='valid', name='maxpool_3')

            body = self.se_conv2d(body, 512, (3, 3), (1, 1), 'same', tf.nn.relu, 'conv2d_4_1', is_train)
            body = self.se_conv2d(body, 512, (3, 3), (1, 1), 'same', tf.nn.relu, 'conv2d_4_2', is_train)
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

