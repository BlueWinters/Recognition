
import tensorflow as tf
import tensorflow.contrib.slim as slim


class MobileNetV2:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    def depthwise_conv2d(self, input, chl_multi, is_train, name):
        with tf.variable_scope(name):
            in_chl = input.get_shape().as_list()[3]
            filter = tf.get_variable('depthwise_weight', shape=(3, 3, in_chl, chl_multi), dtype=tf.float32)
            body = tf.nn.depthwise_conv2d_native(input, filter, (1, 1, 1, 1), 'SAME', name='conv2d')
            return body

    def pointwise_conv2d(self, input, out_chl, strides, padding, is_train, name):
        with tf.variable_scope(name):
            body = tf.layers.conv2d(input, out_chl, (1, 1), strides, padding='valid', name='point_wise')
            return body

    def conv2d_batch_norm_relu(self, input, out_chl, chl_multi, is_train, name):
        with tf.variable_scope(name):
            # depth-wise convolution
            body = self.depthwise_conv2d(input, 1, is_train, 'depthwise_conv2d')
            body = tf.layers.batch_normalization(body, momentum=0.9, training=is_train, name='bn1')
            body = tf.nn.relu(body, name='relu1')
            # point-wise convolution
            body = self.pointwise_conv2d(body, out_chl, (1, 1), 'valid', is_train, 'point_wise')
            body = tf.layers.batch_normalization(body, momentum=0.9, training=is_train, name='bn2')
            body = tf.nn.relu(body, name='relu2')
            return body

    def dropout(self, input, rate, is_train, name):
        return tf.layers.dropout(input, rate=rate, training=is_train, name=name)

    def max_pool(self, input, name):
        return tf.layers.max_pooling2d(input, (2, 2), (2, 2), padding='valid', name=name)

    def forward(self, is_train, reuse, args=None):
        with tf.variable_scope('mobilenet', reuse=reuse):
            body = self.x
            body = self.conv2d_batch_norm_relu(body, 64, 1, is_train, name='block1')
            body = self.dropout(body, 0.3, is_train, name='dropout1')
            body = self.conv2d_batch_norm_relu(body, 128, 1, is_train, name='block2')
            body = self.max_pool(body, name='maxpool1')

            body = self.conv2d_batch_norm_relu(body, 128, 1, is_train, name='block3')
            body = self.dropout(body, 0.4, is_train, name='dropout2')
            body = self.conv2d_batch_norm_relu(body, 128, 1, is_train, name='block4')
            body = self.max_pool(body, name='maxpool2')

            body = self.conv2d_batch_norm_relu(body, 256, 1, is_train, name='block5')
            body = self.dropout(body, 0.4, is_train, name='dropout3')
            body = self.conv2d_batch_norm_relu(body, 256, 1, is_train, name='block6')
            body = self.dropout(body, 0.4, is_train, name='dropout4')
            body = self.conv2d_batch_norm_relu(body, 256, 1, is_train, name='block7')
            body = self.max_pool(body, name='maxpool3')

            body = self.conv2d_batch_norm_relu(body, 512, 1, is_train, name='block8')
            body = self.dropout(body, 0.4, is_train, name='dropout5')
            body = self.conv2d_batch_norm_relu(body, 512, 1, is_train, name='block9')
            body = self.dropout(body, 0.4, is_train, name='dropout6')
            body = self.conv2d_batch_norm_relu(body, 512, 1, is_train, name='block10')
            body = self.max_pool(body, name='maxpool4')

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
