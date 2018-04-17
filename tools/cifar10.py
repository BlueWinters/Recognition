
import numpy as np
import scipy.io as sio
import tools.iterator as iter


class Cifar10:
    def __init__(self, data_path='E:/dataset/cifar10/mat'):
        self.data_path = data_path
        self.package = None

    def preprocess(self, images, labels, data_dim=4, one_hot=True, norm=True):
        if data_dim == 2:
            N = images.shape[0]
            images = np.reshape(images, [N, -1])  # shape: [*,?,?,?] --> [*,?]
        if one_hot == False:
            labels = np.argmax(labels)  # [*,?] --> [?]
        if norm == True:
            images = (images - 127.5) / 127.5
            # images = images / 255.
        return images, labels

    def load_train_data(self, data_dim=4, one_hot=True, norm=True, **kwargs):
        print('Extract train data: [cifar10] from {}'.format(self.data_path))
        data = sio.loadmat('{}/cifar10_train.mat'.format(self.data_path))
        images = data['images'].astype(np.float32)
        labels = data['labels'].astype(np.float32)

        images, labels = self.preprocess(images, labels)
        # package into iterator
        assert self.package == None
        self.package = iter.Iterator(images, labels)

    def load_test_data(self, data_dim=4, one_hot=True, norm=True):
        print('Extract test data: [cifar10] from {}'.format(self.data_path))
        data = sio.loadmat('{}/cifar10_test.mat'.format(self.data_path))
        images = data['images'].astype(np.float32)
        labels = data['labels'].astype(np.float32)

        images, labels = self.preprocess(images, labels)
        # package into iterator
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
