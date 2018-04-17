
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time as time
import imgaug as ia
import zoo.cifar10 as cifar10
from imgaug import augmenters as iaa




class InceptionV1:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    def inception(self, input, num_1x1, num_3x3red, num_3x3, num_5x5red, num_5x5, num_pool, is_train, name):
        with tf.variable_scope(name):
            branch1x1 = tf.layers.conv2d(input, num_1x1, (1, 1), (1, 1), padding='same', activation=tf.nn.relu, name='branch1')

            branch3x3 = tf.layers.conv2d(input, num_3x3red, (1, 1), (1, 1), padding='same', activation=tf.nn.relu, name='branch3x3_1')
            branch3x3 = tf.layers.conv2d(branch3x3, num_3x3, (3, 3), (1, 1), padding='same', activation=tf.nn.relu, name='branch3x3_2')

            branch5x5 = tf.layers.conv2d(input, num_5x5red, (1, 1), (1, 1), padding='same', activation=tf.nn.relu, name='branch5x5_1')
            branch5x5 = tf.layers.conv2d(branch5x5, num_5x5, (5, 5), (1, 1), padding='same', activation=tf.nn.relu, name='branch5x5_2')

            branchpool = tf.layers.max_pooling2d(input, (3, 3), (1, 1), padding='same', name='branchpool_1')
            branchpool = tf.layers.conv2d(branchpool, num_pool, (1, 1), (1, 1), padding='same', activation=tf.nn.relu, name='branchpool_2')

            output = tf.concat([branch1x1, branch3x3, branch5x5, branchpool], axis=3)
            return output

    def forward(self, is_train, reuse):
        with tf.variable_scope('inception_v1', reuse=reuse):
            body = tf.layers.conv2d(self.x, 64, (7, 7), (2, 2), padding='same', activation=tf.nn.relu, name='conv0')
            body = tf.layers.max_pooling2d(body, (3, 3), (2, 2), name='maxpool0')
            body = tf.layers.conv2d(body, 64, (1, 1), (1, 1), padding='same', name='conv1')
            body = tf.layers.conv2d(body, 192, (3, 3), (1, 1), padding='same', name='conv2')
            body = tf.layers.max_pooling2d(body, (3, 3), (2, 2), padding='valid', name='maxpool1')

            body = self.inception(body, 64, 96, 128, 16, 32, 32, is_train, name='inception_1')
            body = self.inception(body, 128, 128, 192, 32, 92, 64, is_train, name='inception_2')
            body = tf.layers.max_pooling2d(body, (3, 3), (2, 2), padding='valid', name='maxpool2')

            body = self.inception(body, 192, 96, 208, 16, 48, 64, is_train, name='inception_1')
            body = self.inception(body, 160, 112, 224, 24, 64, 64, is_train, name='inception_2')
            body = self.inception(body, 128, 128, 256, 24, 64, 64, is_train, name='inception_3')
            body = self.inception(body, 112, 144, 288, 32, 64, 64, is_train, name='inception_4')
            body = self.inception(body, 256, 160, 320, 32, 128, 64, is_train, name='inception_5')
            body = tf.layers.max_pooling2d(body, (3, 3), (2, 2), padding='valid', name='maxpool3')

            body = self.inception(body, 256, 160, 320, 32, 128, 128, is_train, name='inception_6')
            body = self.inception(body, 384, 192, 384, 48, 128, 128, is_train, name='inception_7')

            # global average pool
            body = tf.reduce_mean(body, axis=[1,2])
            # flatten
            body = slim.flatten(body, scope='flatten')

            # output probability
            logits = tf.layers.dense(body, 10, name='logits')
            prediction = tf.nn.softmax(logits)

            # loss
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y))
            reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            weight_decay = 0.0001
            loss = loss + weight_decay * reg_loss
            return loss, prediction
