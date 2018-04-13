
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time as time
import imgaug as ia
import zoo.cifar10 as cifar10
from imgaug import augmenters as iaa




class VggNet:
    def __init__(self):
        self.tiny = 1e-8
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    def build_correct_counter(self, prediction, y):
        correct = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(y, axis=1))
        return tf.reduce_sum(tf.cast(correct, tf.float32))

    def conv2d_batch_norm_relu(self, input, filter, is_train, reuse, name):
        with tf.variable_scope(name, reuse=reuse):
            body = tf.layers.conv2d(input, filter, (3, 3), (1, 1), padding='same', name='conv2d',
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                                    bias_regularizer=tf.contrib.layers.l2_regularizer(0.0001))
            body = tf.layers.batch_normalization(body, momentum=0.5, training=is_train, name='bn')
            body = tf.nn.relu(body, name='relu')
            print(body.get_shape().as_list())
            return body

    def dropout(self, input, rate, is_train, name):
        return tf.layers.dropout(input, rate=rate, training=is_train, name=name)

    def max_pool(self, input, name):
        return tf.layers.max_pooling2d(input, (2, 2), (2, 2), padding='valid', name=name)

    def forward(self, is_train, reuse):
        with tf.variable_scope('vggnet', reuse=reuse):
            body = self.conv2d_batch_norm_relu(self.x, 64, is_train, reuse, name='block1')
            body = self.dropout(body, 0.3, is_train, name='dropout1')
            body = self.conv2d_batch_norm_relu(body, 128, is_train, reuse, name='block2')
            body = self.max_pool(body, name='maxpool1')

            body = self.conv2d_batch_norm_relu(body, 128, is_train, reuse, name='block3')
            body = self.dropout(body, 0.4, is_train, name='dropout2')
            body = self.conv2d_batch_norm_relu(body, 128, is_train, reuse, name='block4')
            body = self.max_pool(body, name='maxpool2')

            body = self.conv2d_batch_norm_relu(body, 256, is_train, reuse, name='block5')
            body = self.dropout(body, 0.4, is_train, name='dropout3')
            body = self.conv2d_batch_norm_relu(body, 256, is_train, reuse, name='block6')
            body = self.dropout(body, 0.4, is_train, name='dropout4')
            body = self.conv2d_batch_norm_relu(body, 256, is_train, reuse, name='block7')
            body = self.max_pool(body, name='maxpool3')

            body = self.conv2d_batch_norm_relu(body, 512, is_train, reuse, name='block8')
            body = self.dropout(body, 0.4, is_train, name='dropout5')
            body = self.conv2d_batch_norm_relu(body, 512, is_train, reuse, name='block9')
            body = self.dropout(body, 0.4, is_train, name='dropout6')
            body = self.conv2d_batch_norm_relu(body, 512, is_train, reuse, name='block10')
            body = self.max_pool(body, name='maxpool4')

            # global average pool
            # body = tf.reduce_mean(body, axis=[1,2])
            # flatten
            body = slim.flatten(body, scope='flatten')

            # output probability
            logits = tf.layers.dense(body, 10, name='logits')
            prediction = tf.nn.softmax(logits)
            correct_counter = self.build_correct_counter(prediction, self.y)

            # loss
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y))
            reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            loss = loss + reg_loss

            return loss, prediction, correct_counter




def train_vggnet_on_cifar10():
    train, test = cifar10.Cifar10(), cifar10.Cifar10()
    train.load_train_data(data_dim=4, one_hot=True, norm=True)
    test.load_test_data(data_dim=4, one_hot=True, norm=True)

    num_epochs = 500*50
    step_epochs = int(num_epochs/100)
    batch_size = 128
    learn_rate = 0.005
    average_loss = 0
    save_path = 'save/vggnet/vgg11_4'

    sess = tf.Session()
    network = VggNet()
    train_loss, _, _ = network.forward(is_train=True, reuse=False)
    eval_loss, pred, correct_counter = network.forward(is_train=False, reuse=True)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        solver = tf.train.AdamOptimizer(learn_rate).minimize(train_loss)

    saver = tf.train.Saver(tf.global_variables())

    sess.run(tf.global_variables_initializer())
    file = open('{}/vgg11_cifar10_train.txt'.format(save_path), 'w')

    # restore and continue train
    # saver.restore(sess, '{}/vgg11_cifar10_model'.format('save/vggnet/vgg_small'))

    # time
    start = time.time()

    seq = iaa.Sequential([iaa.CropAndPad()])

    # training
    for epochs in range(1, num_epochs + 1):
        batch_x, batch_y = train.next_batch(batch_size)
        _, loss = sess.run([solver, train_loss],
                           feed_dict={network.x:batch_x, network.y:batch_y})
        average_loss += loss / step_epochs

        if epochs % step_epochs == 0:
            finish_ratio = int(epochs / step_epochs)
            elapsed = time.time() - start
            acc_counter = sess.run(correct_counter, feed_dict={network.x:batch_x, network.y:batch_y})
            liner = 'epoch {:3d} %, loss {:.6f}, eval {:.6f}, time {:.2f}'\
                .format(finish_ratio, average_loss, float(acc_counter/batch_size), elapsed)
            print(liner), file.writelines(liner + '\n')
            file.flush()
            average_loss, start = 0, time.time()

    # save model
    saver.save(sess, '{}/vgg11_cifar10_model'.format(save_path))

    # evaluate accuracy
    evaluate_batch_size = 100

    evaluate_train_epochs = int(train.num_examples / evaluate_batch_size)
    evaluate_test_epochs = int(test.num_examples / evaluate_batch_size)
    train_acc_counter, train_loss = 0., 0.
    test_acc_counter, test_loss = 0., 0.

    for epochs in range(evaluate_train_epochs):
        batch_x, batch_y = train.next_batch(evaluate_batch_size)
        counter, loss = sess.run([correct_counter, eval_loss],
                                 feed_dict={network.x:batch_x, network.y:batch_y})
        train_acc_counter += counter
        train_loss += loss

    for epochs in range(evaluate_test_epochs):
        batch_x, batch_y = test.next_batch(evaluate_batch_size)
        counter, loss = sess.run([correct_counter, eval_loss],
                                 feed_dict={network.x:batch_x, network.y:batch_y})
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
    network = VggNet()
    train_loss, _, _ = network.forward(is_train=True, reuse=False)
    eval_loss, pred, correct_counter = network.forward(is_train=False, reuse=True)