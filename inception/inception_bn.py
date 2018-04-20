
import tensorflow as tf
import tensorflow.contrib.slim as slim


class InceptionBN:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    def conv2d_bn_relu(self, input, out_chl, kernel, stride, padding, is_train, name=None):
        with tf.variable_scope(name):
            body = tf.layers.conv2d(input, out_chl, kernel, stride, padding=padding, name='conv2d')
            body = tf.layers.batch_normalization(body, momentum=0.9, training=is_train, name='bn')
            body = tf.nn.relu(body, name='relu')
            return body

    def inception_bn(self, input, num_1x1, num_3x3, is_train, name):
        with tf.variable_scope(name):
            branch1x1 = tf.layers.conv2d(input, num_1x1, (1, 1), (1, 1), padding='valid', name='conv1')
            branch1x1 = tf.layers.batch_normalization(branch1x1, momentum=0.9, training=is_train, name='bn1')
            branch1x1 = tf.nn.relu(branch1x1, name='relu1')

            branch3x3 = tf.layers.conv2d(input, num_3x3, (3, 3), (1, 1), padding='same', name='conv2')
            branch3x3 = tf.layers.batch_normalization(branch3x3, momentum=0.9, training=is_train, name='bn2')
            branch3x3 = tf.nn.relu(branch3x3, name='relu2')

            return tf.concat([branch1x1, branch3x3], axis=3)

    def inception_downsample(self, input, num_3x3, is_train, name):
        with tf.variable_scope(name):
            branch3x3 = tf.layers.conv2d(input, num_3x3, (3, 3), (2, 2), padding='same', name='conv2d')
            branch3x3 = tf.layers.batch_normalization(branch3x3, momentum=0.9, training=is_train, name='bn')
            branch3x3 = tf.nn.relu(branch3x3, name='relu')

            branchpool = tf.layers.max_pooling2d(input, (3, 3), (2, 2), 'same', name='max_pool')

            return tf.concat([branch3x3, branchpool], axis=3)

    def forward(self, args, is_train, reuse):
        with tf.variable_scope('inception_bn', reuse=reuse):
            # input
            body = self.x
            body = self.conv2d_bn_relu(body, 96, (3, 3), (1, 1), 'same', is_train, 'conv2d')
            #
            body = self.inception_bn(body, 32, 48, is_train, 'inception3a')
            body = self.inception_bn(body, 32, 48, is_train, 'inception3b')
            body = self.inception_downsample(body, 80, is_train, 'inception3c')

            body = self.inception_bn(body, 112, 48, is_train, 'inception4a')
            body = self.inception_bn(body, 96, 64, is_train, 'inception4b')
            body = self.inception_downsample(body, 96, is_train, 'inception4c')
            body = self.inception_bn(body, 80, 80, is_train, 'inception4d')
            body = self.inception_bn(body, 48, 96, is_train, 'inception4e')
            body = self.inception_downsample(body, 96, is_train, 'inception4f')

            body = self.inception_bn(body, 176, 160, is_train, 'inception5a')
            body = self.inception_bn(body, 176, 160, is_train, 'inception5b')
            body = self.inception_downsample(body, 128, is_train, 'inception5c')

            # global average pool
            body = tf.reduce_mean(body, axis=[1,2])
            # flatten
            body = slim.flatten(body, scope='flatten')

            # output probability
            logits = tf.layers.dense(body, 10, name='logits')
            prediction = tf.nn.softmax(logits)

            # loss
            l2_vars = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y))
            weight_decay = 0.0001
            loss = weight_decay * l2_vars + cross_entropy

            return loss, prediction

