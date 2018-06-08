
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Iterator(object):
	def __init__(self, images, labels):
		self.images = images
		self.labels = labels
		self.num_examples = images.shape[0]
		self.epochs_completed = 0
		self.index_in_epoch = 0

	def next_batch(self, batch_size, shuffle=True):
		start = self.index_in_epoch

		# Shuffle for the first epoch
		if self.epochs_completed == 0 and start == 0 and shuffle:
			perm0 = np.arange(self.num_examples)
			np.random.shuffle(perm0)
			self.images = self.images[perm0]
			self.labels = self.labels[perm0]

		# Go to the next epoch
		if start + batch_size > self.num_examples:
			# Finished epoch
			self.epochs_completed += 1
			# Get the rest examples in this epoch
			rest_num_examples = self.num_examples - start
			images_rest_part = self.images[start:self.num_examples]
			labels_rest_part = self.labels[start:self.num_examples]
			# Shuffle the data
			if shuffle:
				perm = np.arange(self.num_examples)
				np.random.shuffle(perm)
				self.images = self.images[perm]
				self.labels = self.labels[perm]
			# Start next epoch
			start = 0
			self.index_in_epoch = batch_size - rest_num_examples
			end = self.index_in_epoch
			images_new_part = self.images[start:end]
			labels_new_part = self.labels[start:end]
			return np.concatenate((images_rest_part, images_new_part), axis=0), \
				   np.concatenate((labels_rest_part, labels_new_part), axis=0)
		else:
			self.index_in_epoch += batch_size
			end = self.index_in_epoch
			return self.images[start:end], self.labels[start:end]

	def is_iter_over(self, batch_size):
		return (self.index_in_epoch + batch_size) > self.num_examples

	def reset(self):
		self.index_in_epoch = 0
		self.epochs_completed = 0
