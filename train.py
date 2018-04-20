
import tensorflow as tf
import time as time
import argparse
import shutil
import os
import tools.cifar10 as cifar10
import tools.helper as helper

import vggnet.vggnet as vgg_sr
import vggnet.vggnet_bn as vgg_bn
import vggnet.vggnet_se as vgg_se

import resnet.resnet as resnet
import resnet.resnext as resnext
import densenet.densenet as densenet
import inception.inception_bn as inception_bn
import dpnet.dpnet as dpnet

def train_on_cifar10(args):
    def evaluation_on_test(evaluate_batch_size=100):
        assert test.num_examples % evaluate_batch_size == 0
        evaluate_test_epochs = int(test.num_examples / evaluate_batch_size)
        test_acc_top1, test_acc_top3, test_loss = 0., 0., 0.
        for epochs in range(evaluate_test_epochs):
            batch_x, batch_y = test.next_batch(evaluate_batch_size)
            t1, t3, loss = sess.run([top1, top3, eval_loss],
                                    feed_dict={network.x: batch_x, network.y: batch_y})
            test_acc_top1 += t1
            test_acc_top3 += t3
            test_loss += loss
        test_acc_top1 = float(test_acc_top1 / test.num_examples)
        test_acc_top3 = float(test_acc_top3 / test.num_examples)
        test_loss = float(test_loss / test.num_examples)
        return test_acc_top1, test_acc_top3, test_loss

    def build_top_k(predictions, labels, k, name):
        with tf.name_scope(name):
            target = tf.cast(tf.argmax(labels, axis=1), tf.int32)
            in_top_k = tf.to_float(tf.nn.in_top_k(predictions, target, k=k))
            return tf.reduce_sum(tf.cast(in_top_k, tf.float32))

    def get_model_symbol(model_name, args):
        if model_name == 'vggnet':
            return vgg_sr.VggNet()
        elif model_name == 'vggnet_bn':
            return vgg_bn.VggNet()
        elif model_name == 'vggnet_se':
            return vgg_se.VggNet()
        elif model_name == 'resnext':
            return resnext.ResNeXt()
        elif model_name == 'resnet':
            return resnet.ResNet()
        elif model_name == 'densenet':
            return densenet.DenseNet()
        elif model_name == 'inception_bn':
            return inception_bn.InceptionBN()
        elif model_name == 'dpnet':
            return dpnet.DPNet()
        else:
            raise ValueError('no such model [{}]'.format(model_name))

    def get_data_set_symbol(data_set):
        train, test = cifar10.Cifar10(), cifar10.Cifar10()
        train.load_train_data(data_dim=4, one_hot=True, norm=True)
        test.load_test_data(data_dim=4, one_hot=True, norm=True)
        return train, test

    def save_training_config(args):
        file = open('{}/config'.format(args.save_path), 'w')
        command = vars(args)
        for key in command:
            file.write('{}={}\n'.format(key, command[key]))
        file.close()
        # save scheduler file
        _, lr_sr_file_name = os.path.split(args.lr_scheduler)
        lr_dst_file_path = '{}/{}'.format(args.save_path, lr_sr_file_name)
        shutil.copyfile(args.lr_scheduler, lr_dst_file_path)

    # save train config
    save_training_config(args=args)

    model_name = args.model_name
    data_set = args.data_set
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    save_path = args.save_path
    lr_scheduler = args.lr_scheduler

    train, test = get_data_set_symbol(data_set=data_set)
    network = get_model_symbol(model_name=model_name, args=args)
    train_loss, train_pred = network.forward(is_train=True, reuse=None, args=args)
    eval_loss, test_pred = network.forward(is_train=False, reuse=True, args=args)

    top1 = build_top_k(predictions=test_pred, labels=network.y, k=1, name='top1')
    top3 = build_top_k(predictions=test_pred, labels=network.y, k=3, name='top3')

    learn_rate = tf.placeholder(dtype=tf.float32, name='lr')
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        solver = tf.train.GradientDescentOptimizer(learn_rate).minimize(train_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    file = open('{}/{}_{}_train.txt'.format(save_path, model_name, data_set), 'w')
    scheduler = helper.lr_scheduler(lr_scheduler)
    saver = tf.train.Saver(tf.global_variables())
    lr = 0.

    # training
    for epochs in range(1, num_epochs + 1):
        num_iter = int(train.num_examples // batch_size)
        average_loss, start = 0, time.time()
        lr = scheduler[epochs] if epochs in scheduler else lr

        for iter in range(num_iter):
            batch_x, batch_y = train.next_batch(batch_size)
            _, loss = sess.run([solver, train_loss],
                               feed_dict={network.x:batch_x, network.y:batch_y, learn_rate:lr})
            average_loss += loss / num_iter

        elapsed = time.time() - start
        # acc1, acc3, test_loss = sess.run([top1, top3, eval_loss], feed_dict={network.x:batch_x, network.y:batch_y})
        acc1, acc3, test_loss = evaluation_on_test(evaluate_batch_size=100)
        liner = 'epoch {:3d}, train loss {:.6f}, eval loss {:.6f}, eval top_1 {:.6f}, eval top_3 {:.6f}, time {:.2f}' \
            .format(epochs, average_loss, test_loss, acc1, acc3, elapsed)
        print(liner), file.writelines(liner + '\n')
        file.flush()


    # save model
    saver.save(sess, '{}/{}_{}_model'.format(save_path, model_name, data_set))

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



# inception_bn
# resnext
# densenet


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command for training cifar10")
    parser.add_argument('--model_name', type=str, default='dpnet', help='')
    parser.add_argument('--data_set', type=str, default='cifar10', help='')
    parser.add_argument('--save_path', type=str, default='tmp/dpnet', help='the directory to save model')
    parser.add_argument('--batch_size', type=int, default=64, help='')
    parser.add_argument('--num_epochs', type=int, default=1, help='')
    parser.add_argument('--lr_scheduler', type=str, default='tools/lr_scheduler', help='learning rate scheduler')

    # parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')

    # for resnet/resnext
    # parser.add_argument('--num_blocks', type=list, default=[8,8,8], help='for resnet/resnext')
    # parser.add_argument('--chl_list', type=list, default=[16,16,32,64], help='for resnet/resnext')

    # for densenet
    # parser.add_argument('--num_blocks', type=int, default=1, help='for densenet')
    # parser.add_argument('--num_layers', type=int, default=16, help='for densenet')
    # parser.add_argument('--growth_chl', type=int, default=12, help='for densenet')


    args = parser.parse_args()
    train_on_cifar10(args)