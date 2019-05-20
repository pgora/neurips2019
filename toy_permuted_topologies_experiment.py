# this file contains code for a permuted graph topologies experiment
# (we permute columns and rows of the adjacency matrix and check the
# resulting neural networks fits)

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
	numbers_of_transpositions = list(range(22))
	no_of_samples = 100
	numbers_of_transpositions *= no_of_samples
	file_to_save_results = "toy_permuted_topologies_0.csv"
	with open(file_to_save_results,"a") as f:
		for i in numbers_of_transpositions:
			print("Constructing random adjacency matrix with %d transpositions" % i)
			random_permutation = graph_utils.generate_random_permutation_with_approx_no_of_transpositions(21, i)
			random_adj_matrix = ochota_adj_matrix[random_permutation][:,random_permutation]
			symmetric_diff = graph_utils.undirected_symmetric_difference(random_adj_matrix, ochota_adj_matrix)
			tf.reset_default_graph()
			nn_input = tf.placeholder(dtype=tf.float32, shape=[None, 21])
			targets = tf.placeholder(dtype=tf.float32, shape=[None, 1])
			print("Constructing graph neural net")
			nn_output = graph_nn.transfer_matrix_neural_net(nn_input, 3, 4, tf.nn.tanh, random_adj_matrix, verbose=False)
			optimizer = tf.train.AdamOptimizer(0.005)
			batch_iterator = data_prep.BatchIterator(X_train, y_train, 997)
			print("Training network with symmetric diff %d" % symmetric_diff)
			test_and_batch_losses = train_eval.train_model(nn_output, nn_input, targets, optimizer, 100000, batch_iterator, X_test, y_test, "trained_networks/toy_permuted_model_tmp.ckpt", 1000, verbose=True)
			test_loss = test_and_batch_losses[-1][0]
			model_avg_error, actual_vs_predicted = train_eval.evaluate_model_on_a_dataset("trained_networks/toy_permuted_model_tmp.ckpt", nn_output,nn_input, X_test, y_test, y_scaler)
			test_loss = test_and_batch_losses[-1][0]
			f.write("%d,%f\n" % (i, test_loss))
			f.flush()
			print((i, symmetric_diff, test_loss))
			# old version with error, symmetric diff missing here
			#print((i, test_loss))
			# old version, no good for this toy problem:
			#f.write("%d,%d,%f,%f\n" % (i, symmetric_diff, model_avg_error, test_loss))
			#f.flush()
			#print((symmetric_diff, model_avg_error, test_loss))
		f.close()
