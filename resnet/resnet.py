
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time as time
import cifar10 as cifar10
import tools as tl




class ResNet:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 3), name='x')
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, 10), name='y')
        self.lr = tf.placeholder(dtype=tf.float32, name='lr')

    def residual_unit(self, input, out_chl, bottle_neck, chl_increase, stride, name, is_train):
        # ref: https://github.com/tornadomeet/ResNet/blob/master/symbol_resnet.py
        with tf.variable_scope(name):
            in_chl = input.get_shape().as_list()[-1]
            if bottle_neck == True:
                # body = tf.layers.batch_normalization(input, momentum=0.9, training=is_train, name='bn1')
                # body = tf.nn.relu(body, name='relu1')
                # body = tf.layers.conv2d(body, out_chl, (1, 1), (1, 1), 'same', use_bias=False, name='conv1')
                #
                # body = tf.layers.batch_normalization(body, momentum=0.9, training=is_train, name='bn2')
                # body = tf.nn.relu(body, name='relu2')
                # body = tf.layers.conv2d(body, out_chl, (3, 3), stride, 'same', use_bias=False, name='conv2')
                #
                # body = tf.layers.batch_normalization(body, momentum=0.9, training=is_train, name='bn3')
                # body = tf.nn.relu(body, name='relu3')
                # body = tf.layers.conv2d(body, out_chl, (1, 1), (1, 1), 'same', use_bias=False, name='conv3')
                #
                # if in_chl != out_chl or stride != (1, 1):
                #     shortcut = tf.layers.conv2d(input, out_chl, (1, 1), stride, 'same', use_bias=False, name='shortcut')
                # else:
                #     shortcut = input
                # return body + shortcut
                pass
            else:
                body = tf.layers.batch_normalization(input, momentum=0.9, training=is_train, name='bn1')
                body = short = tf.nn.relu(body, name='relu1')
                body = tf.layers.conv2d(body, out_chl, (3, 3), (1, 1), 'same', use_bias=False, name='conv1')
                body = tf.layers.dropout(body, rate=0.4, training=is_train, name='drop1')

                body = tf.layers.batch_normalization(body, momentum=0.9, training=is_train, name='bn2')
                body = tf.nn.relu(body, name='relu2')
                body = tf.layers.conv2d(body, out_chl, (3, 3), stride, 'same', use_bias=False, name='conv2')
                body = tf.layers.dropout(body, rate=0.4, training=is_train, name='drop2')

                if chl_increase == True:
                    if stride == (1, 1):
                        shortcut = tf.layers.conv2d(input, out_chl, (1, 1), stride, 'same', use_bias=False, name='shortcut')
                    else:
                        chl = (out_chl - in_chl) // 2
                        shortcut = tf.layers.max_pooling2d(input, stride, stride, padding='valid', name='max_pool')
                        shortcut = tf.pad(shortcut, [[0, 0], [0, 0], [0, 0], [chl, chl]], name='pad')
                else:
                    shortcut = input

                output = body + shortcut
                return output

    def build_correct_counter(self, prediction, y):
        correct = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(y, axis=1))
        return tf.reduce_sum(tf.cast(correct, tf.float32))

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

    def build_resnet(self, num_blocks, chl_list, bottle_neck, is_train, reuse):
        # config
        num_stage = len(num_blocks)
        if is_train == True:
            self.print_graph(num_blocks, chl_list, bottle_neck)

        # build graph
        with tf.variable_scope('resnet', reuse=reuse):
            # input batch normalization
            body = tf.layers.conv2d(self.x, chl_list[0], (3, 3), (1, 1), 'same', use_bias=True, name='conv0')

            for i in range(num_stage):
                name = 'stage{}_unit{}'.format(i+1, 1)
                stride = (1 if i == 0 else 2,) * 2 # a tuple, stride like (1,1) or (2,2)
                body = self.residual_unit(body, chl_list[i+1], False, True, stride, name, is_train)
                for j in range(num_blocks[i]-1):
                    name = 'stage{}_unit{}'.format(i+1, j+2)
                    body = self.residual_unit(body, chl_list[i+1], False, False, (1, 1), name, is_train)

            body = tf.layers.batch_normalization(body, momentum=0.9, training=is_train, name='bn1')
            body = tf.nn.relu(body, name='relu1')

            # body = tf.reduce_mean(body, axis=[1, 2], name='gap')
            body = tf.layers.average_pooling2d(body, (7, 7), (7, 7), name='avg_pool')
            body = slim.flatten(body, scope='flatten')
            logit = tf.layers.dense(body, 10, name='fc')
            prediction = tf.nn.softmax(logit, name='softmax')

            l2_vars = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=self.y))
            weight_decay = 0.0001
            loss = weight_decay * l2_vars + cross_entropy
            correct_counter = self.build_correct_counter(prediction, self.y)

            return loss, prediction, correct_counter



def train_resnet_on_cifar10():
    def evalute_on_test(evaluate_batch_size=100):
        evaluate_test_epochs = int(test.num_examples / evaluate_batch_size)
        test_acc_counter, test_loss = 0., 0.
        for epochs in range(evaluate_test_epochs):
            batch_x, batch_y = test.next_batch(evaluate_batch_size)
            counter, loss = sess.run([correct_counter, eval_loss],
                                     feed_dict={network.x: batch_x, network.y: batch_y})
            test_acc_counter += counter
            test_loss += loss
        test_acc = float(test_acc_counter / test.num_examples)
        test_loss = float(test_loss / test.num_examples)
        return float(test_acc_counter/test.num_examples)

    # data
    train, test = cifar10.Cifar10(), cifar10.Cifar10()
    train.load_train_data(data_dim=4, one_hot=True)
    test.load_test_data(data_dim=4, one_hot=True)

    # training config
    num_epochs = 50
    step_epochs = 100#int(num_epochs/100)
    batch_size = 64
    learn_rate = 0.
    save_path = 'save/v1'

    # network config
    chl_list = [16, 16, 32, 64]
    num_blocks = [8, 8, 8]
    bottle_neck = False

    network = ResNet()
    train_loss, _, _ = network.build_resnet(num_blocks, chl_list, bottle_neck, is_train=True, reuse=None)
    eval_loss, pred, correct_counter = network.build_resnet(num_blocks, chl_list, bottle_neck, is_train=False, reuse=True)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        solver = tf.train.GradientDescentOptimizer(network.lr).minimize(train_loss)


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    file = open('{}/resnet_cifar10_train.txt'.format(save_path), 'w')
    scheduler = tl.lr_scheduler('lr')

    saver = tf.train.Saver(tf.global_variables())
    # saver.restore(sess, 'save/v1/tmp2/resnet_cifar10_model')


    # training
    for epochs in range(1, num_epochs+1):
        num_iter = int(train.num_examples // batch_size)
        average_loss, start = 0, time.time()
        learn_rate = scheduler[epochs] if epochs in scheduler else learn_rate

        for iter in range(num_iter):
            batch_x, batch_y = train.next_batch(batch_size)
            _, loss = sess.run([solver, train_loss],
                               feed_dict={network.x:batch_x, network.y:batch_y, network.lr:learn_rate})
            average_loss += loss / step_epochs

        # output information
        elapsed = time.time() - start
        # acc_counter = sess.run(correct_counter, feed_dict={network.x:batch_x, network.y:batch_y})
        test_acc = evalute_on_test(evaluate_batch_size=10)
        liner = 'epoch {:d}/{:d}, loss {:.6f}, eval {:.6f}, time {:.2f}' \
            .format(epochs, num_epochs, average_loss, test_acc, elapsed)
        print(liner), file.writelines(liner + '\n')
        file.flush()


    # save model
    saver.save(sess, '{}/resnet_cifar10_model'.format(save_path))

    evaluate_batch_size = 10
    evaluate_train_epochs = int(train.num_examples / evaluate_batch_size)
    evaluate_test_epochs = int(test.num_examples / evaluate_batch_size)
    train_acc_counter, train_loss = 0., 0.
    test_acc_counter, test_loss = 0., 0.

    for epochs in range(evaluate_train_epochs):
        batch_x, batch_y = train.next_batch(evaluate_batch_size)
        counter, loss = sess.run([correct_counter, eval_loss],
                                 feed_dict={network.x: batch_x, network.y: batch_y})
        train_acc_counter += counter
        train_loss += loss

    for epochs in range(evaluate_test_epochs):
        batch_x, batch_y = test.next_batch(evaluate_batch_size)
        counter, loss = sess.run([correct_counter, eval_loss],
                                 feed_dict={network.x: batch_x, network.y: batch_y})
        test_acc_counter += counter
        test_loss += loss

    train_acc = float(train_acc_counter / train.num_examples)
    test_acc = float(test_acc_counter / test.num_examples)
    train_loss = float(train_loss / train.num_examples)
    test_loss = float(test_loss / test.num_examples)
    liner = 'train: accuracy {:.4f} loss {:.6f}\n' \
            'test: accuracy {:.4f} loss {:.6f}'.format(train_acc, train_loss, test_acc, test_loss)
    print(liner), file.writelines(liner + '\n')

    sess.close()
    file.close()


if __name__ == '__main__':
    train_resnet_on_cifar10()
    # depth = 50
    # per_unit = [int((depth - 2) / 6)]
    # chl_list = [16, 16, 32, 64]
    # bottle_neck = False
    # units = per_unit * 3
    # network = ResNet()
    # train_loss, _, _ = network.build_resnet(units, 3, chl_list, bottle_neck, is_train=True, reuse=None)