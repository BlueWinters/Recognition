
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
        self.reg = tf.contrib.layers.l2_regularizer(0.0001)

    def build_correct_counter(self, prediction, y):
        correct = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(y, axis=1))
        return tf.reduce_sum(tf.cast(correct, tf.float32))

    def conv2d_bn_relu(self, input, num_chl, kernel, stride, padding, is_train, name=None):
        with tf.variable_scope(name):
            body = tf.layers.conv2d(input, num_chl, kernel, stride, padding=padding, name='conv2d')
            body = tf.layers.batch_normalization(body, momentum=0.9, training=is_train, name='bn')
            body = tf.nn.relu(body, name='relu')
            return body

    def inception_bn_A(self, input, num_1x1, num_3x3red, num_3x3, num_5x5red, num_5x5, num_pool, is_train, name):
        with tf.variable_scope(name):
            branch1x1 = self.conv2d_bn_relu(input, num_1x1, (1, 1), (1, 1), 'valid', is_train, 'branch1x1')

            branch3x3 = self.conv2d_bn_relu(input, num_3x3red, (1, 1), (1, 1), 'valid', is_train, 'branch3x3_1')
            branch3x3 = self.conv2d_bn_relu(branch3x3, num_3x3, (3, 3), (1, 1), 'same', is_train, 'branch3x3_2')

            branch5x5 = self.conv2d_bn_relu(input, num_5x5red, (1, 1), (1, 1), 'valid', is_train, 'branch3x3_1')
            branch5x5 = self.conv2d_bn_relu(branch5x5, num_5x5, (3, 3), (1, 1), 'same', is_train, 'branch3x3_2')
            branch5x5 = self.conv2d_bn_relu(branch5x5, num_5x5, (3, 3), (1, 1), 'same', is_train, 'branch3x3_3')

            branchpool = tf.layers.max_pooling2d(input, (3, 3), (1, 1), 'same', name='max_pool')
            branchpool = self.conv2d_bn_relu(branchpool, num_pool, (1, 1), (1, 1), 'valid', is_train, 'branchpool')

            output = tf.concat([branch1x1, branch3x3, branch5x5, branchpool], axis=3)
            return output

    def inception_bn_B(self, input, num_3x3red, num_3x3, num_5x5red, num_5x5, is_train, name):
        with tf.variable_scope(name):
            branch3x3 = self.conv2d_bn_relu(input, num_3x3red, (1, 1), (1, 1), 'valid', is_train, 'branch3x3_1')
            branch3x3 = self.conv2d_bn_relu(branch3x3, num_3x3, (3, 3), (2, 2), 'same', is_train, 'branch3x3_2')

            branch5x5 = self.conv2d_bn_relu(input, num_5x5red, (1, 1), (1, 1), 'valid', is_train, 'branch3x3_1')
            branch5x5 = self.conv2d_bn_relu(branch5x5, num_5x5, (3, 3), (1, 1), 'same', is_train, 'branch3x3_2')
            branch5x5 = self.conv2d_bn_relu(branch5x5, num_5x5, (3, 3), (2, 2), 'same', is_train, 'branch3x3_3')

            branchpool = tf.layers.max_pooling2d(input, (3, 3), (2, 2), 'same', name='max_pool')

            output = tf.concat([branch3x3, branch5x5, branchpool], axis=3)
            return output

    def forward_imagenet(self, input, is_train, reuse):
        with tf.variable_scope('inception_bn', reuse=reuse):
            # stage 1
            body = self.conv2d_bn_relu(input, 64, (7, 7), (2, 2), 'same', name='stage1')
            body = tf.layers.max_pooling2d(body, (3, 3), (2, 2), padding='valid', name='stage1_maxpool')
            # stage 2
            body = self.conv2d_bn_relu(body, 64, (1, 1), (1, 1), 'same', name='stage2_1')
            body = self.conv2d_bn_relu(body, 192, (3, 3), (1, 1), 'same', name='stage2_2')
            body = tf.layers.max_pooling2d(body, (3, 3), (2, 2), padding='valid', name='stage2_maxpool')
            # stage 3
            body = self.inception_bn_A(body, 64, 64, 64, 64, 96, 32, is_train, 'stage3_a')
            body = self.inception_bn_A(body, 64, 64, 96, 64, 96, 64, is_train, 'stage3_b')
            body = self.inception_bn_B(body, 128, 160, 64, 96, is_train, 'stage3_c')
            # stage 4
            body = self.inception_bn_A(body, 224, 64, 96, 96, 128, 128, is_train, 'stage4_a')
            body = self.inception_bn_A(body, 192, 96, 128, 96, 128, 128, is_train, 'stage4_b')
            body = self.inception_bn_A(body, 160, 128, 160, 128, 160, 128, is_train, 'stage4_c')
            body = self.inception_bn_A(body, 96, 128, 192, 160, 192, 128, is_train, 'stage4_d')
            body = self.inception_bn_B(body, 128, 192, 192, 256, is_train, 'stage4_a')
            # stage 5
            body = self.inception_bn_A(body, 352, 192, 320, 160, 224, 128, is_train, 'stage5_a')
            body = self.inception_bn_A(body, 352, 192, 320, 192, 224, 128, is_train, 'stage5_b')

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
