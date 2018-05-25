
import tensorflow as tf
import tensorflow.contrib.slim as slim


class VggNet:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        self.ratio = 8

    def se_conv2d_batch_norm_relu(self, input, out_chl, is_train, name):
        with tf.variable_scope(name):
            ratio = self.ratio
            assert out_chl % ratio == 0
            body = tf.layers.conv2d(input, out_chl, (3, 3), (1, 1), 'same', name='conv2d')
            body = tf.layers.batch_normalization(body, momentum=0.9, training=is_train, name='bn')
            body = tf.nn.relu(body, name='relu')
            with tf.name_scope('squeeze_excitation'):
                squeeze = tf.reduce_mean(body, axis=(1, 2), name='squeeze')
                excitation = tf.layers.dense(squeeze, units=int(out_chl/ratio), activation=tf.nn.relu, name='dense1')
                excitation = tf.layers.dense(excitation, units=int(out_chl), activation=tf.nn.sigmoid, name='dense2')
                excitation = tf.reshape(excitation, shape=(-1, 1, 1, out_chl))
                output = body * excitation
            return output

    def dropout(self, input, rate, is_train, name):
        return tf.layers.dropout(input, rate=rate, training=is_train, name=name)

    def max_pool(self, input, name):
        return tf.layers.max_pooling2d(input, (2, 2), (2, 2), padding='valid', name=name)

    def forward(self, is_train, reuse, args=None):
        with tf.variable_scope('vggnet', reuse=reuse):
            body = self.x
            body = self.se_conv2d_batch_norm_relu(body, 64, is_train, name='block1')
            body = self.dropout(body, 0.3, is_train, name='dropout1')
            body = self.se_conv2d_batch_norm_relu(body, 128, is_train, name='block2')
            body = self.max_pool(body, name='maxpool1')

            body = self.se_conv2d_batch_norm_relu(body, 128, is_train, name='block3')
            body = self.dropout(body, 0.4, is_train, name='dropout2')
            body = self.se_conv2d_batch_norm_relu(body, 128, is_train, name='block4')
            body = self.max_pool(body, name='maxpool2')

            body = self.se_conv2d_batch_norm_relu(body, 256, is_train, name='block5')
            body = self.dropout(body, 0.4, is_train, name='dropout3')
            body = self.se_conv2d_batch_norm_relu(body, 256, is_train, name='block6')
            body = self.dropout(body, 0.4, is_train, name='dropout4')
            body = self.se_conv2d_batch_norm_relu(body, 256, is_train, name='block7')
            body = self.max_pool(body, name='maxpool3')

            body = self.se_conv2d_batch_norm_relu(body, 512, is_train, name='block8')
            body = self.dropout(body, 0.4, is_train, name='dropout5')
            body = self.se_conv2d_batch_norm_relu(body, 512, is_train, name='block9')
            body = self.dropout(body, 0.4, is_train, name='dropout6')
            body = self.se_conv2d_batch_norm_relu(body, 512, is_train, name='block10')
            body = self.max_pool(body, name='maxpool4')

            # max-min pooling
            max_body = tf.layers.max_pooling2d(body, (2, 2), (2, 2), name='gap_max_pool')
            min_body = tf.layers.average_pooling2d(body, (2, 2), (2, 2), name='gap_avg_pool')
            body = tf.concat([min_body, max_body], axis=3, name='mix')
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
