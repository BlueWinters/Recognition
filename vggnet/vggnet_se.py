
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time as time
import tools.tools as tl
import tools.cifar10 as cifar10




class VggNet:
    def __init__(self):
        self.tiny = 1e-8
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        self.reg = tf.contrib.layers.l2_regularizer(0.0001)
        self.r = 8

    def build_top_k(self, predictions, labels, k, name):
        with tf.name_scope(name):
            in_top_k = tf.to_float(tf.nn.in_top_k(predictions, tf.argmax(labels, axis=1), k=k))
            return tf.reduce_sum(tf.cast(in_top_k, tf.float32))

    def se_conv2d(self, input, out_chl, kernel, stride, padding, activation, name):
        with tf.variable_scope(name):
            assert out_chl % self.r == 0
            body = tf.layers.conv2d(input, out_chl, kernel, stride, padding=padding, activation=activation, name='conv2d')
            squeeze = tf.reduce_mean(body, axis=(1, 2), name='squeeze')
            excitation = tf.layers.dense(squeeze, units=int(out_chl/self.r), activation=activation, name='dense1')
            excitation = tf.layers.dense(excitation, units=int(out_chl), activation=tf.nn.sigmoid, name='dense2')
            return body * excitation

    def forward(self, is_train, reuse):
        with tf.variable_scope('vggnet', reuse=reuse):
            body = self.x
            body = self.se_conv2d(body, 64, (3, 3), (1, 1), 'same', tf.nn.relu, 'conv2d_1_1')
            body = self.se_conv2d(body, 128, (3, 3), (1, 1), 'same', tf.nn.relu, 'conv2d_1_2')
            body = tf.layers.max_pooling2d(body, (2, 2), (2, 2), padding='same', name='maxpool_1')

            body = self.se_conv2d(body, 256, (3, 3), (1, 1), 'same', tf.nn.relu, 'conv2d_2_1')
            body = self.se_conv2d(body, 256, (3, 3), (1, 1), 'same', tf.nn.relu, 'conv2d_2_2')
            body = tf.layers.max_pooling2d(body, (2, 2), (2, 2), padding='same', name='maxpool_2')

            body = self.se_conv2d(body, 512, (3, 3), (1, 1), 'same', tf.nn.relu, 'conv2d_3_1')
            body = self.se_conv2d(body, 512, (3, 3), (1, 1), 'same', tf.nn.relu, 'conv2d_3_1')
            body = tf.layers.max_pooling2d(body, (2, 2), (2, 2), padding='same', name='maxpool_3')

            body = self.se_conv2d(body, 512, (3, 3), (1, 1), 'same', tf.nn.relu, 'conv2d_4_1')
            body = self.se_conv2d(body, 512, (3, 3), (1, 1), 'same', tf.nn.relu, 'conv2d_4_2')
            body = tf.layers.max_pooling2d(body, (2, 2), (2, 2), padding='same', name='maxpool_4')

            # flatten
            body = slim.flatten(body, scope='flatten')
            # output probability
            logit = tf.layers.dense(body, 10, name='logits')
            prediction = tf.nn.softmax(logit)
            top_1_counter = self.build_top_k(prediction, self.y, k=1, name='top1')
            top_3_counter = self.build_top_k(prediction, self.y, k=3, name='top3')

            # loss
            l2_vars = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=self.y))
            weight_decay = 0.0001
            loss = weight_decay * l2_vars + cross_entropy

            return loss, prediction, top_1_counter, top_3_counter



def train_vggnet_on_cifar10():
    def evalute_on_test(evaluate_batch_size=100):
        evaluate_test_epochs = int(test.num_examples / evaluate_batch_size)
        test_acc_counter, test_loss = 0., 0.
        for epochs in range(evaluate_test_epochs):
            batch_x, batch_y = test.next_batch(evaluate_batch_size)
            counter, loss = sess.run([top1, eval_loss],
                                     feed_dict={network.x: batch_x, network.y: batch_y})
            test_acc_counter += counter
            test_loss += loss
        test_acc = float(test_acc_counter / test.num_examples)
        test_loss = float(test_loss / test.num_examples)
        return float(test_acc_counter/test.num_examples)

    train, test = cifar10.Cifar10(), cifar10.Cifar10()
    train.load_train_data(data_dim=4, one_hot=True, norm=True)
    test.load_test_data(data_dim=4, one_hot=True, norm=True)

    num_epochs = 100
    batch_size = 64
    learn_rate = 0.
    save_path = 'save/vggnet/vggnet_se'

    network = VggNet()
    train_loss, _, _, _ = network.forward(is_train=True, reuse=False)
    eval_loss, pred, top1, top3 = network.forward(is_train=False, reuse=True)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        solver = tf.train.AdamOptimizer(learn_rate).minimize(train_loss)


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    file = open('{}/vggnet_se_cifar10_train.txt'.format(save_path), 'w')
    scheduler = tl.lr_scheduler('lr')

    saver = tf.train.Saver(tf.global_variables())
    # saver.restore(sess, '{}/vgg11_cifar10_model'.format('save/vggnet/vgg_small'))


    # training
    for epochs in range(1, num_epochs + 1):
        num_iter = int(train.num_examples // batch_size)
        average_loss, start = 0, time.time()
        learn_rate = scheduler[epochs] if epochs in scheduler else learn_rate

        for iter in range(num_iter):
            batch_x, batch_y = train.next_batch(batch_size)
            _, loss = sess.run([solver, train_loss],
                               feed_dict={network.x:batch_x, network.y:batch_y})
            average_loss += loss / num_iter

        elapsed = time.time() - start
        acc1, acc3 = sess.run([top1, top3], feed_dict={network.x:batch_x, network.y:batch_y})
        liner = 'epoch {:3d} %, loss {:.6f}, eval top_1 {:.6f}, eval top_3 {:.6f}, time {:.2f}' \
            .format(epochs, average_loss, float(acc1/batch_size), float(acc3/batch_size), elapsed)
        print(liner), file.writelines(liner + '\n')
        file.flush()


    # save model
    saver.save(sess, '{}/vggnet_se_cifar10_model'.format(save_path))

    # evaluate accuracy
    evaluate_batch_size = 100

    evaluate_train_epochs = int(train.num_examples / evaluate_batch_size)
    evaluate_test_epochs = int(test.num_examples / evaluate_batch_size)
    train_acc1_counter, train_acc3_counter, train_loss = 0., 0., 0.
    test_acc1_counter, test_acc3_counter, test_loss = 0., 0., 0.

    for epochs in range(evaluate_train_epochs):
        batch_x, batch_y = train.next_batch(evaluate_batch_size)
        counter1, counter3, loss = sess.run([top1, top3, eval_loss],
                                 feed_dict={network.x:batch_x, network.y:batch_y})
        train_acc1_counter += counter1
        train_acc3_counter += counter3
        train_loss += loss

    for epochs in range(evaluate_test_epochs):
        batch_x, batch_y = test.next_batch(evaluate_batch_size)
        counter1, counter3, loss = sess.run([top1, top3, eval_loss],
                                 feed_dict={network.x:batch_x, network.y:batch_y})
        test_acc1_counter += counter1
        test_acc3_counter += counter3
        test_loss += loss

    train_acc1 = float(train_acc1_counter / train.num_examples)
    train_acc3 = float(train_acc3_counter / train.num_examples)
    test_acc1 = float(test_acc1_counter / test.num_examples)
    test_acc3 = float(test_acc3_counter / test.num_examples)
    train_loss = float(train_loss / train.num_examples)
    test_loss = float(test_loss / test.num_examples)
    liner = 'train: accuracy top1 {:.4f}, top3 {:.4f}, loss {:.6f}\n' \
            'test: accuracy top1 {:.4f}, top3 {:.4f} loss {:.6f}'.format(train_acc1, train_acc3, train_loss,
                                                                         test_acc1, test_acc3, test_loss)
    print(liner), file.writelines(liner + '\n')

    sess.close()
    file.close()


if __name__ == '__main__':
    pass
    # network = VggNet()
    # train_loss, _, _ = network.forward(is_train=True, reuse=False)
    # eval_loss, pred, correct_counter = network.forward(is_train=False, reuse=True)