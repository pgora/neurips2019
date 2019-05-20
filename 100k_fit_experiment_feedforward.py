# this file contains code for running 100k set fit and evaluate experiments from command line - the fit part

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

# this function constructs a simple feedforward neural net with the specified number of hidden layers and neurons
def simple_feedforward_neural_net(inpt, no_of_layers, no_of_neurons, activation_function):
    hidden = tf.layers.dense(inpt, no_of_neurons, activation=activation_function, kernel_initializer=tf.contrib.layers.xavier_initializer())
    for i in range(no_of_layers - 1):
        hidden = tf.layers.dense(hidden, no_of_neurons, activation=activation_function, kernel_initializer=tf.contrib.layers.xavier_initializer())
    return tf.layers.dense(hidden, 1, kernel_initializer=tf.contrib.layers.xavier_initializer())

layers = [2,3,4]
neurons = [20,40,100]
activations = [tf.nn.tanh]

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

if __name__ == '__main__':
	ochota_adj_matrix = np.genfromtxt("macierz_sasiedztwa.txt")
	print("Loaded adjacency matrix")
	traffic_lights_data = pd.read_csv("100k.csv", header=None)
	print("Read datafile")
	X, y, X_scaler, y_scaler = data_prep.scale_standard_traffic_light_data(traffic_lights_data)
	print("Scaled the data")
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=831191)

	# variable for storing job numbers
	i = 0

	for no_of_layers in layers:
		for no_of_neurons in neurons:
			for activation in activations:
				i += 1
				print("Reseting default graph")
				tf.reset_default_graph()

				nn_input = tf.placeholder(dtype=tf.float32, shape=[None, 21])
				targets = tf.placeholder(dtype=tf.float32, shape=[None, 1])

				if activation == tf.nn.relu:
					activation_name = "relu"
				else:
					activation_name = "tanh"

				model_checkpoint_file = "100k_feedforward_fit_and_evaluate_experiments/model_%d_%d_%s.ckpt" % (no_of_layers, no_of_neurons, activation_name)
				print("Constructing network with %d layers, %d neurons per layer and %s activation function" % (no_of_layers, no_of_neurons, activation_name))
				#nn_output = graph_nn.transfer_matrix_neural_net(nn_input, no_of_layers, no_of_channels, activation, ochota_adj_matrix, verbose=True, share_weights_in_transfer_matrix=False, share_biases_in_transfer_matrix=False)
				# this time we're constructing simple feedforward neural net:
				nn_output = simple_feedforward_neural_net(nn_input, no_of_layers, no_of_neurons, activation)
				print("Creating optimizer")
				optimizer = tf.train.AdamOptimizer(0.0035)
				print("Creating batch iterator")
				batch_iterator = data_prep.BatchIterator(X_train, y_train, 997)
				print("Training and saving the network")
				test_and_batch_losses = train_eval.train_model(nn_output, nn_input, targets, optimizer, 100000, batch_iterator, X_test, y_test, model_checkpoint_file, 1000, verbose=True)
				job_file_name = "100k_feedforward_fit_and_evaluate_experiments/%d.job" % i
				with open(job_file_name,"w") as f:
					f.write("%d,%d,%s,%s" % (no_of_layers, no_of_neurons, activation_name, model_checkpoint_file))
					print("Saved a job as %s" % job_file_name)
					f.close()


