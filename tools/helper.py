
import matplotlib.pyplot as plt
import numpy as np



def lr_scheduler(file_name):
    scheduler = {}
    file = open(file_name, 'r')
    for liner in file:
        info = liner.split(',')
        epoch = int(info[0])
        lr = float(info[1])
        scheduler[epoch] = lr
    return scheduler

def split_train_txt(full_file_path):
    loss, eval = [], []
    top1, top3 = [], []
    file = open(full_file_path, 'r')
    for liner in file:
        if 'epoch' in liner:
            all = liner.split(',')
            loss.append(float(all[1].split(' ')[-1]))
            eval.append(float(all[2].split(' ')[-1]))
            top1.append(float(all[3].split(' ')[-1]))
            top3.append(float(all[4].split(' ')[-1]))
    file.close()

    np_loss = np.array(loss, np.float32)
    np_eval = np.array(eval, np.float32)
    np_top1 = 1 - np.array(top1, np.float32)
    np_top3 = 1 - np.array(top3, np.float32)
    return np_loss, np_eval, np_top1, np_top3

def plot_vggnet_performance(number, name):
    vggnet_val = split_train_txt('E:/ComputerVision/Recognition/save/without_aug/vggnet/vggnet_cifar10_train.txt')
    vggnet_bn = split_train_txt('E:/ComputerVision/Recognition/save/without_aug/vggnet_bn/vggnet_bn_cifar10_train.txt')
    vggnet_pool = split_train_txt('E:/ComputerVision/Recognition/save/without_aug/vggnet_pool/vggnet_pool_cifar10_train.txt')
    vggnet_se = split_train_txt('E:/ComputerVision/Recognition/save/without_aug/vggnet_se/vggnet_se_cifar10_train.txt')

    x = np.array(range(100))
    plt.plot(x, vggnet_val[number])
    plt.plot(x, vggnet_bn[number])
    plt.plot(x, vggnet_pool[number])
    plt.plot(x, vggnet_se[number])
    plt.title(name)
    plt.legend(('vggnet', 'vggnet_bn', 'vggnet_pool', 'vggnet_se'), loc='upper right')
    plt.show()


if __name__ == '__main__':
    plot_vggnet_performance(2, 'accuracy')