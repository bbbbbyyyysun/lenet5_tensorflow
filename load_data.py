from tensorflow.examples.tutorials.mnist import input_data
import numpy as np 
from sklearn.utils import shuffle

def load_data():
	mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
	X_train, y_train           = mnist.train.images, mnist.train.labels
	X_validation, y_validation = mnist.validation.images, mnist.validation.labels
	X_test, y_test             = mnist.test.images, mnist.test.labels


	assert(len(X_train) == len(y_train))
	assert(len(X_validation) == len(y_validation))
	assert(len(X_test) == len(y_test))

	# convert the size from 28*28*1 to 32*32*1
	X_train = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
	X_validation = np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
	X_test = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')

	X_train, y_train = shuffle(X_train, y_train)

	return X_train, y_train, X_validation, y_validation, X_test, y_test