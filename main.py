from lenet5 import lenet5
from load_data import load_data
import argparse
import tensorflow as tf 
import cv2
import numpy as np 

def train():
	X_train, y_train, X_validation, y_validation, X_test, y_test = load_data()
	net = lenet5(X_train, y_train, X_validation, y_validation, X_test, y_test)
	net.train(epoches=50, batch_size=100)

def test():
	sess = tf.Session()
	# retrieve meta graph and restore weights
	saver = tf.train.import_meta_graph('./tmp/model.meta')
	saver.restore(sess, tf.train.latest_checkpoint('./tmp/'))

	# retrive the input and output tensors from graph
	graph = tf.get_default_graph()
	x = graph.get_tensor_by_name('x:0')
	logits = graph.get_tensor_by_name('logits:0')
	pred = tf.argmax(logits, 1)

	img = get_img()

	if img == 'quit': 
		return

	img = cv2.resize(img, (32,32))
	img = extractValue(img)
	img = np.array(img)
	img = img[np.newaxis,:,:,np.newaxis]
	
	result = sess.run(pred, feed_dict={x:img})[0]
	print('The prediction is {}'.format(result))
	cv2.imshow('img', img)
	cv2.waitKey(0)

def get_img():
	while True:
		s = input('Input image filename (press q to quit):')
		if s == 'q':
			return 'quit'

		try:
			img = cv2.imread(s)
			return img
		except:
			print('Incorrect filename! Try again!')
			continue
	return

def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape

    imgHSV = np.zeros((height, width, 3), np.uint8)

    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)

    return imgValue

if __name__ == '__main__':
	parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
	# model 0 is train; model 1 is test
	parser.add_argument('-model', type=int)
	
	Flages = parser.parse_args()

	if Flages.model == 0:
		train()
	elif Flages.model == 1:
		test()
	else:
		print('Invalid flage info!')
