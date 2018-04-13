
import matplotlib.pyplot as plt
import numpy as np


def plot_curve():
    save_path = 'save/v3/1/resnet_cifar10_train.txt'
    loss = []
    eval = []
    file = open(save_path, 'r')
    for liner in file:
        if 'epoch' in liner:
            all = liner.split(',')
            str_loss = all[1]
            str_eval = all[2]
            loss.append(float(str_loss.split(' ')[-1]))
            eval.append(float(str_eval.split(' ')[-1]))
    file.close()


    x = np.array(range(len(loss)))

    y_loss = np.array(loss)
    axe1 = plt.subplot(121)
    axe1.plot(x, y_loss)

    y_eval = np.array(eval)
    axe2 = plt.subplot(122)
    axe2.plot(x, y_eval)

    plt.show()

def lr_scheduler(file_name):
    scheduler = {}
    file = open(file_name, 'r')
    for liner in file:
        info = liner.split(',')
        epoch = int(info[0])
        lr = float(info[1])
        scheduler[epoch] = lr
    return scheduler

def lr_config(epoch, scheduler):
    if epoch in scheduler:
        return scheduler[epoch]
    else:
        return None
