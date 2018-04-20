
import tensorflow as tf
import tensorflow.contrib.slim as slim



class DenseNet:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, (None, 32, 32, 3), 'x')
        self.y = tf.placeholder(tf.float32, (None, 10), 'y')

    def block_layer(self, input, num_filter, is_train, name):
        with tf.variable_scope(name):
            body = tf.layers.batch_normalization(input, momentum=0.9, training=is_train, name='bn')
            body = tf.nn.relu(body, name='relu')
            body = tf.layers.conv2d(body, num_filter, (3, 3), (1, 1), 'same', use_bias=False, name='con2d')
            # body = tf.layers.dropout(body, rate=0.2, training=is_train, name='dropout')
            return body

    def denset_block(self, input, in_chl, num_layers, growth_chl, is_train, name):
        with tf.variable_scope(name):
            body = input
            for n in range(num_layers):
                add_body = self.block_layer(body, growth_chl, is_train, 'layer{}'.format(n))
                body = tf.concat([body, add_body], 3, name='concat{}'.format(n))
                in_chl += growth_chl
            return body, in_chl

    def transition(self, input, out_chl, is_train, name):
        with tf.variable_scope(name):
            body = tf.layers.batch_normalization(input, momentum=0.9, training=is_train, name='bn')
            body = tf.nn.relu(body, name='relu')
            body = tf.layers.conv2d(body, out_chl, (1, 1), (1, 1), 'same', use_bias=False, name='conv2d')
            body = tf.layers.dropout(body, rate=0.5, training=is_train, name='dropout')
            body = tf.layers.average_pooling2d(body, (2, 2), (2, 2), 'valid', name='avg_pool')
            return body

    def forward(self, args, is_train, reuse):
        num_blocks = 4#args.num_blocks
        num_layers = 8#args.num_layers
        growth_chl = 16#args.growth_chl

        with tf.variable_scope('densenet', reuse=reuse):
            body = tf.layers.conv2d(self.x, 16, (3, 3), (1, 1), 'same', name='conv')
            cur_chl = 16

            for n in range(num_blocks):
                body, cur_chl = self.denset_block(body, cur_chl, num_layers, growth_chl,
                                                  is_train, 'block{}'.format(n))
                if n < num_blocks-1:
                    body = self.transition(body, cur_chl, is_train, 'transition{}'.format(n))

            body = tf.layers.batch_normalization(body, momentum=0.9, training=is_train, name='bn')
            body = tf.nn.relu(body, name='relu')

            body = tf.reduce_mean(body, axis=(1, 2), name='gap')
            body = slim.flatten(body, scope='flatten')
            logit = tf.layers.dense(body, 10, name='dense')
            prediction = tf.nn.softmax(logit, name='softmax')

            l2_vars = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=self.y))
            weight_decay = 0.0001
            loss = weight_decay * l2_vars + cross_entropy

            return loss, prediction

