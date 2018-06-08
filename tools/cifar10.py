
import numpy as np
import scipy.io as sio
import tools.iterator as iter


class Cifar10:
    def __init__(self, data_path='E:/dataset/cifar10/mat', is_pixel=None):
        self.data_path = data_path
        self.package = None

    def load_train_data(self):
        print('Extract train data: [cifar10] from {}'.format(self.data_path))
        data = sio.loadmat('{}/cifar10_train.mat'.format(self.data_path))

        images = data['images'].astype(np.float32)
        labels = data['labels'].astype(np.float32)
        assert self.package == None
        self.package = iter.Iterator(images, labels)

    def load_test_data(self):
        print('Extract test data: [cifar10] from {}'.format(self.data_path))
        data = sio.loadmat('{}/cifar10_test.mat'.format(self.data_path))

        images = data['images'].astype(np.float32)
        labels = data['labels'].astype(np.float32)

        assert self.package == None
        self.package = iter.Iterator(images, labels)

    def next_batch(self, batch_size, shuffle=True):
        return self.package.next_batch(batch_size=batch_size, shuffle=shuffle)

    @property
    def num_examples(self):
        return self.package.num_examples

    @property
    def images(self):
        return self.package.images

    @property
    def labels(self):
        return self.package.labels
