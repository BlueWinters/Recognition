
import tensorflow as tf
import tensorflow.contrib.slim as slim



class ResNet:
    def __init__(self):
        self.bottle_neck = False
        self.block_size = [8, 8, 8]
        self.num_filters = [16, 32, 64]
        self.block_stride = (1, 2, 2)
        self.batch_norm_decay = 0.997
        self.batch_norm_epsilon = 1e-5
        self.kernel_initializer = tf.variance_scaling_initializer

    def residual_unit(self, input, out_chl, bottle_neck, chl_increase, stride, name, is_train):
        # https://github.com/tornadomeet/ResNet/blob/master/symbol_resnet.py
        # https://github.com/ppwwyyxx/tensorpack/blob/master/examples/ResNet/cifar10-resnet.py
        with tf.variable_scope(name):
            in_chl = input.get_shape().as_list()[-1]
            if bottle_neck == True:
                body = tf.layers.batch_normalization(input, momentum=self.batch_norm_decay,
                    epsilon=self.batch_norm_epsilon, training=is_train, name='bn1')
                body = tf.nn.relu(body, name='relu1')
                body = tf.layers.conv2d(body, out_chl, (1, 1), (1, 1), 'same', use_bias=False,
                    kernel_initializer=self.kernel_initializer, name='conv1')

                body = tf.layers.batch_normalization(body, momentum=self.batch_norm_decay,
                    epsilon=self.batch_norm_epsilon, training=is_train, name='bn2')
                body = tf.nn.relu(body, name='relu2')
                body = tf.layers.conv2d(body, out_chl, (3, 3), stride, 'same', use_bias=False,
                    kernel_initializer=self.kernel_initializer, name='conv2')

                body = tf.layers.batch_normalization(body, momentum=self.batch_norm_decay,
                    epsilon=self.batch_norm_epsilon, training=is_train, name='bn3')
                body = tf.nn.relu(body, name='relu3')
                body = tf.layers.conv2d(body, out_chl, (1, 1), (1, 1), 'same', use_bias=False,
                    kernel_initializer=self.kernel_initializer, name='conv3')

                if chl_increase == True:
                    if stride == (1, 1):
                        shortcut = tf.layers.conv2d(input, out_chl, (1, 1), (1, 1), 'same', use_bias=False,
                            kernel_initializer=self.kernel_initializer, name='short_conv2d')
                    else:
                        chl = (out_chl - in_chl) // 2
                        shortcut = tf.layers.max_pooling2d(input, (2, 2), (2, 2), padding='valid', name='max_pool')
                        shortcut = tf.pad(shortcut, [[0, 0], [0, 0], [0, 0], [chl, chl]], name='pad')
                else:
                    shortcut = input
                return body + shortcut
            else:
                body = tf.layers.batch_normalization(input, momentum=self.batch_norm_decay,
                    epsilon=self.batch_norm_epsilon, training=is_train, name='bn1')
                body = tf.nn.relu(body, name='relu1')
                body = tf.layers.conv2d(body, out_chl, (3, 3), stride, 'same', use_bias=False,
                    kernel_initializer=self.kernel_initializer, name='conv1')

                body = tf.layers.batch_normalization(body, momentum=self.batch_norm_decay,
                    epsilon=self.batch_norm_epsilon, training=is_train, name='bn2')
                body = tf.nn.relu(body, name='relu2')
                body = tf.layers.conv2d(body, out_chl, (3, 3), (1, 1), 'same', use_bias=False,
                    kernel_initializer=self.kernel_initializer, name='conv2')

                if chl_increase == True:
                    # the first block use stride (1, 1), then the others use (2, 2)
                    if stride == (1, 1):
                        shortcut = tf.layers.conv2d(input, out_chl, (1, 1), (1, 1), 'same', use_bias=False,
                            kernel_initializer=self.kernel_initializer, name='short_conv2d')
                    else:
                        chl = (out_chl - in_chl) // 2
                        shortcut = tf.layers.max_pooling2d(input, (2, 2), (2, 2), padding='valid', name='max_pool')
                        shortcut = tf.pad(shortcut, [[0, 0], [0, 0], [0, 0], [chl, chl]], name='pad')
                else:
                    shortcut = input

                output = body + shortcut
                return output

    def build_graph(self, is_train, reuse=None, args=None):
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 3), name='x')
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, 10), name='y')

        # build graph
        with tf.variable_scope('resnet_v2', reuse=reuse):
            # pre-conv2d
            body = tf.layers.conv2d(self.x, 16, (3, 3), (1, 1), 'same', use_bias=False,
                kernel_initializer=self.kernel_initializer, name='pre_conv')

            for i, num_block in enumerate(self.block_size):
                name = 'block{}_unit{}'.format(i, 0)
                stride = (self.block_stride[i], ) * 2
                body = self.residual_unit(body, self.num_filters[i], self.bottle_neck, True, stride, name, is_train)
                for j in range(1, num_block):
                    name = 'block{}_unit{}'.format(i, j)
                    body = self.residual_unit(body, self.num_filters[i], self.bottle_neck, False, (1, 1), name, is_train)

            # post batch norm & relu
            body = tf.layers.batch_normalization(body, momentum=self.batch_norm_decay,
                epsilon=self.batch_norm_epsilon, training=is_train, name='post_bn')
            body = tf.nn.relu(body, name='post_relu')

            body = tf.reduce_mean(body, axis=[1, 2], name='gap')
            body = slim.flatten(body, scope='flatten')
            logit = tf.layers.dense(body, 10, name='fc')
            prediction = tf.nn.softmax(logit, name='softmax')

            l2_vars = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=self.y))
            weight_decay = 0.0001
            loss = weight_decay * l2_vars + cross_entropy

            return loss, prediction



if __name__ == '__main__':
    network = ResNet()
    network.build_graph(True)