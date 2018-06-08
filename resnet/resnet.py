
import tensorflow as tf
import tensorflow.contrib.slim as slim



class ResNet:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 3), name='x')
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, 10), name='y')

    def residual_unit(self, input, out_chl, bottle_neck, chl_increase, stride, name, is_train):
        # https://github.com/tornadomeet/ResNet/blob/master/symbol_resnet.py
        # https://github.com/ppwwyyxx/tensorpack/blob/master/examples/ResNet/cifar10-resnet.py
        with tf.variable_scope(name):
            in_chl = input.get_shape().as_list()[-1]
            if bottle_neck == True:
                body = tf.layers.batch_normalization(input, momentum=0.9, training=is_train, name='bn1')
                body = tf.nn.relu(body, name='relu1')
                body = tf.layers.conv2d(body, out_chl, (1, 1), (1, 1), 'same', use_bias=False, name='conv1')
                body = tf.layers.dropout(body, rate=0.2, training=is_train, name='drop1')

                body = tf.layers.batch_normalization(body, momentum=0.9, training=is_train, name='bn2')
                body = tf.nn.relu(body, name='relu2')
                body = tf.layers.conv2d(body, out_chl, (3, 3), stride, 'same', use_bias=False, name='conv2')
                body = tf.layers.dropout(body, rate=0.2, training=is_train, name='drop2')

                body = tf.layers.batch_normalization(body, momentum=0.9, training=is_train, name='bn3')
                body = tf.nn.relu(body, name='relu3')
                body = tf.layers.conv2d(body, out_chl, (1, 1), (1, 1), 'same', use_bias=False, name='conv3')
                body = tf.layers.dropout(body, rate=0.2, training=is_train, name='drop3')

                if chl_increase == True:
                    if stride == (1, 1):
                        shortcut = tf.layers.conv2d(input, out_chl, (1, 1), (1, 1), 'same', use_bias=False, name='short_conv2d')
                        shortcut = tf.layers.dropout(shortcut, rate=0.2, training=is_train, name='short_dropout')
                    else:
                        chl = (out_chl - in_chl) // 2
                        shortcut = tf.layers.max_pooling2d(input, (2, 2), (2, 2), padding='valid', name='max_pool')
                        shortcut = tf.pad(shortcut, [[0, 0], [0, 0], [0, 0], [chl, chl]], name='pad')
                else:
                    shortcut = input
                return body + shortcut
                pass
            else:
                body = tf.layers.batch_normalization(input, momentum=0.9, training=is_train, name='bn1')
                body = tf.nn.relu(body, name='relu1')
                body = tf.layers.conv2d(body, out_chl, (3, 3), stride, 'same', use_bias=False, name='conv1')
                # body = tf.layers.dropout(body, rate=0.2, training=is_train, name='drop1')

                body = tf.layers.batch_normalization(body, momentum=0.9, training=is_train, name='bn2')
                body = tf.nn.relu(body, name='relu2')
                body = tf.layers.conv2d(body, out_chl, (3, 3), (1, 1), 'same', use_bias=False, name='conv2')
                body = tf.layers.dropout(body, rate=0.2, training=is_train, name='drop2')

                if chl_increase == True:
                    if stride == (1, 1):
                        shortcut = tf.layers.conv2d(input, out_chl, (1, 1), (1, 1), 'same', use_bias=False, name='short_conv2d')
                    else:
                        chl = (out_chl - in_chl) // 2
                        shortcut = tf.layers.average_pooling2d(input, (2, 2), (2, 2), padding='valid', name='ave_pool')
                        shortcut = tf.pad(shortcut, [[0, 0], [0, 0], [0, 0], [chl, chl]], name='pad')
                else:
                    shortcut = input

                output = body + shortcut
                return output

    def print_graph(self, num_blocks, chl_list, bottle_neck):
        # assert
        assert len(num_blocks) == len(num_blocks) - 1
        # print network config
        depth = 2
        for n in num_blocks:
            cur_depth = int(num_blocks[n]) * (6 if bottle_neck is False else 9)
            depth += cur_depth
            print('stage {}, blocks {}, depth {}, channel {}'.format(n + 1, num_blocks[n+1], cur_depth, chl_list[n+1]))
        print('depth {}'.format(depth))

    def forward(self, is_train, reuse, args=None):
        bottle_neck = False
        num_blocks = [8, 8, 8] #args.num_blocks
        chl_list = [16, 32, 64] #args.chl_list
        num_stage = len(num_blocks)

        # build graph
        with tf.variable_scope('resnet', reuse=reuse):
            # input batch normalization
            body = tf.layers.conv2d(self.x, 16, (3, 3), (1, 1), 'same', use_bias=False, name='conv0')

            for i in range(num_stage):
                name = 'stage{}_unit{}'.format(i+1, 1)
                stride = (1 if i == 0 else 2,) * 2 # a tuple, stride like (1,1) or (2,2)
                body = self.residual_unit(body, chl_list[i], bottle_neck, True, stride, name, is_train)
                for j in range(num_blocks[i]-1):
                    name = 'stage{}_unit{}'.format(i+1, j+2)
                    body = self.residual_unit(body, chl_list[i], bottle_neck, False, (1, 1), name, is_train)

            body = tf.layers.batch_normalization(body, momentum=0.9, training=is_train, name='bn1')
            body = tf.nn.relu(body, name='relu1')

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