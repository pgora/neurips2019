# this file contains code for a random graph topologies experiment

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

import os

import graph_utils as graph_utils
import graph_neural_networks as graph_nn
import data_preparation_utils as data_prep
from iterative_updaters import VanillaGradientDescent, MomentumGradientDescent, NesterovMomentumGradientDescent, RMSPropGradientDescent, AdamGradientDescent
import training_and_evaluation as train_eval
import graph_nn_experiments as experiments

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # uncomment this when GPU is supposed to be used

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

if __name__ == '__main__':
	ochota_adj_matrix = np.genfromtxt("macierz_sasiedztwa.txt")
	print("Loaded adjacency matrix")
	toy_data = pd.read_csv("toy_set.csv", header=None)
	print("Read datafile")
	X, y, X_scaler, y_scaler = data_prep.scale_standard_traffic_light_data(toy_data)
	print("Scaled the data")
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=831191)
	print("Divided the data into train/test sets")
	symmetric_diffs = [3 * i for i in range(15)]
	no_of_samples = 100
	symmetric_diffs *= no_of_samples
	file_to_save_results = "toy_random_topologies_0.csv"
	with open(file_to_save_results,"a") as f:
		for i in symmetric_diffs:
			print("Constructing random adjacency matrix with symmetric diff %d" % i)
			random_adj_matrix = graph_utils.generate_random_adjacency_matrix_with_fixed_symmetric_difference(i, ochota_adj_matrix)
			tf.reset_default_graph()
			nn_input = tf.placeholder(dtype=tf.float32, shape=[None, 21])
			targets = tf.placeholder(dtype=tf.float32, shape=[None, 1])
			print("Constructing graph neural net")
			nn_output = graph_nn.transfer_matrix_neural_net(nn_input, 3, 4, tf.nn.tanh, random_adj_matrix, verbose=False, share_weights_in_transfer_matrix=False, share_biases_in_transfer_matrix=False)
			optimizer = tf.train.AdamOptimizer(0.005)
			batch_iterator = data_prep.BatchIterator(X_train, y_train, 997)
			print("Training network with symmetric diff %d" % i)
			test_and_batch_losses = train_eval.train_model(nn_output, nn_input, targets, optimizer, 100000, batch_iterator, X_test, y_test, "trained_networks/toy_random_model_tmp.ckpt", 1000, verbose=True)
			test_loss = test_and_batch_losses[-1][0]
			f.write("%d,%f\n" % (i, test_loss))
			f.flush()
			print((i, test_loss))
		f.close()
