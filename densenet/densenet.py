
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time as time
import cifar10 as cifar10
import tools as tl



class DenseNet:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, (None, 32, 32, 3), 'x')
        self.y = tf.placeholder(tf.float32, (None, 10), 'y')
        self.lr = tf.placeholder(tf.float32, name='lr')

    def correct_counter(self, prediction, y):
        correct = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(y, axis=1))
        return tf.reduce_sum(tf.cast(correct, tf.float32))

    def block_layer(self, input, num_filter, is_train, name):
        with tf.variable_scope(name):
            body = tf.layers.batch_normalization(input, momentum=0.9, training=is_train, name='bn')
            body = tf.nn.relu(body, name='relu')
            body = tf.layers.conv2d(body, num_filter, (3, 3), (1, 1), 'same', use_bias=False, name='con2d')
            body = tf.layers.dropout(body, rate=0.2, training=is_train, name='dropout')
            return body

    def denset_block(self, input, in_chl, num_layers, growth_chl, is_train, name):
        with tf.variable_scope(name):
            feat_list = [input]
            body = input
            for n in range(num_layers):
                body = self.block_layer(body, growth_chl, is_train, 'layer{}'.format(n))
                feat_list.append(body)
                body = tf.concat(feat_list, 3, name='concat{}'.format(n))
                in_chl += growth_chl
            return body, in_chl

    def transition(self, input, out_chl, is_train, name):
        with tf.variable_scope(name):
            body = tf.layers.batch_normalization(input, momentum=0.9, training=is_train, name='bn')
            body = tf.nn.relu(body, name='relu')
            body = tf.layers.conv2d(body, out_chl, (1, 1), (1, 1), 'same', use_bias=False, name='conv2d')
            body = tf.layers.dropout(body, rate=0.2, training=is_train, name='dropout')
            body = tf.layers.average_pooling2d(body, (2, 2), (2, 2), 'valid', name='avg_pool')
            return body

    def build_graph(self, num_blocks, num_layers, growth_chl, is_train, reuse):
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
            correct_counter = self.correct_counter(prediction, self.y)

            return loss, prediction, correct_counter



def train_densenet_on_cifar10():
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

    train, test = cifar10.Cifar10(), cifar10.Cifar10()
    train.load_train_data(data_dim=4, one_hot=True)
    test.load_test_data(data_dim=4, one_hot=True)

    num_epochs = 200
    batch_size = 64
    learn_rate = 0.
    save_path = 'save'

    num_block = 5
    num_layers = 7
    growth_chl = 12
    network = DenseNet()
    train_loss, _, _ = network.build_graph(num_block, num_layers, growth_chl, is_train=True, reuse=None)
    eval_loss, pred, correct_counter = network.build_graph(num_block, num_layers, growth_chl, is_train=False, reuse=True)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        solver = tf.train.MomentumOptimizer(network.lr, momentum=0.9).minimize(train_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    file = open('{}/densenet_cifar10_train.txt'.format(save_path), 'a')
    scheduler = tl.lr_scheduler('lr')

    saver = tf.train.Saver(tf.global_variables())
    # saver.restore(sess, 'save/v5/densenet_cifar10_model')


    # training
    for epochs in range(1, num_epochs+1):
        num_iter = int(train.num_examples // batch_size)
        average_loss, start = 0, time.time()
        learn_rate = scheduler[epochs] if epochs in scheduler else learn_rate

        for iter in range(num_iter):
            batch_x, batch_y = train.next_batch(batch_size)
            _, loss = sess.run([solver, train_loss],
                               feed_dict={network.x:batch_x, network.y:batch_y, network.lr:learn_rate})
            average_loss += loss / num_iter

        # output information
        elapsed = time.time() - start
        # acc_counter = sess.run(correct_counter, feed_dict={network.x:batch_x, network.y:batch_y})
        test_acc = evalute_on_test(evaluate_batch_size=10)
        liner = 'epoch {:d}/{:d}, loss {:.6f}, eval {:.6f}, time {:.2f}' \
            .format(epochs, num_epochs, average_loss, test_acc, elapsed)
        print(liner), file.writelines(liner + '\n')
        file.flush()


    # save model
    saver.save(sess, '{}/densenet_cifar10_model'.format(save_path))

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
    train_densenet_on_cifar10()
