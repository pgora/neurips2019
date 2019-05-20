# this file contains code for running 100k set fit and evaluate experiments from command line - the evaluate part

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

import time
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

working_path = "100k_feedforward_fit_and_evaluate_experiments/"

simulator_microservice_url = 'http://3.122.113.135:25041/'

# set this equal to the number of threads in the simulator virtual machine
no_of_threads = 72

result_file_name = "100k_feedforward_fit_and_evaluate_experiments/fit_eval_results.csv"

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

	
	while(True):
		time.sleep(2)
		print("Checking for jobs")
		jobs = [os.path.join(working_path, file) for file in os.listdir(working_path) if os.path.isfile(os.path.join(working_path, file)) and file.endswith(".job")]

		if len(jobs) > 0:
			job = jobs[0]
			print("Processing job %s" % job)

			with open(job, "r") as f:
				no_of_layers, no_of_neurons, activation_name, model_checkpoint_file = f.readline().split(",")
				no_of_layers = int(no_of_layers)
				no_of_neurons = int(no_of_neurons)
				if activation_name == "relu":
					activation = tf.nn.relu
				else:
					activation = tf.nn.tanh
				f.close()

			
			print("Reseting default graph")
			tf.reset_default_graph()

			nn_input = tf.placeholder(dtype=tf.float32, shape=[None, 21])
			targets = tf.placeholder(dtype=tf.float32, shape=[None, 1])
			print("Constructing network with %d layers, %d neurons per layer and %s activation function" % (no_of_layers, no_of_neurons, activation_name))
			#nn_output = graph_nn.transfer_matrix_neural_net(nn_input, no_of_layers, no_of_channels, activation, ochota_adj_matrix, verbose=True, share_weights_in_transfer_matrix=False, share_biases_in_transfer_matrix=False)
			# this time we're using simple feedforward neural nets
			nn_output = simple_feedforward_neural_net(nn_input, no_of_layers, no_of_neurons, activation)
			print("Restoring network weights from %s and evaluating on test set" % model_checkpoint_file)
			model_avg_error, actual_vs_predicted = train_eval.evaluate_model_on_a_dataset(model_checkpoint_file, nn_output,nn_input, X_test, y_test, y_scaler)
			print("Model avg. error on test set: %f" % model_avg_error)
			# close session (if open)
			try:
				sess.close()
			except:
				pass
			# open new session
			sess =  tf.Session()
			saver = tf.train.Saver()
			saver.restore(sess, model_checkpoint_file)
			# select the number of gradient descent trajectories
			no_of_trajectories = 100
			# select the updater
			#updater = VanillaGradientDescent()
			#updater = MomentumGradientDescent()
			updater = NesterovMomentumGradientDescent()
			#updater = RMSPropGradientDescent(lr=0.01) # <-- there may be something wrong with this one
			#updater = AdamGradientDescent(lr=1.0) # <-- there may be something wrong with this one
			print("Generating gradient descent trajectories")
			trajectories = train_eval.generate_and_join_multiple_gradient_descent_trajectories(sess, no_of_trajectories, nn_output, nn_input, X_scaler, y_scaler, updater, 3000, 30, verbose=True, trajectories_verbose=False)
			print("Generating a test set from gradient descent trajectories")
			simulation_test_X, simulation_test_y = train_eval.generate_test_set_from_trajectory_points(trajectories, no_of_threads, simulator_microservice_url, verbose=True)
			# simulation test set needs to be scaled before plugging it into a network
			simulation_test_Xy = np.concatenate((simulation_test_X, simulation_test_y.reshape(-1,1)), axis=1)
			simulation_test_X = X_scaler.transform(simulation_test_X)
			simulation_test_y = y_scaler.transform(simulation_test_y.reshape(-1,1)).reshape(-1)
			print("Evaluating neural net on the newly create test set")
			simulation_model_avg_error, simulation_actual_vs_predicted = train_eval.evaluate_model_on_a_dataset(model_checkpoint_file, nn_output, nn_input, simulation_test_X, simulation_test_y, y_scaler)
			print("Model avg. error on simulation test set: %f" % simulation_model_avg_error)
			simulation_test_Xy_filename = "100k_feedforward_fit_and_evaluate_experiments/simulation_Xy_%d_%d_%s.csv" % (no_of_layers, no_of_neurons, activation_name)
			print("Saving simulation test set to file")
			np.savetxt(simulation_test_Xy_filename, simulation_test_Xy, delimiter=",")
			print("Saving results to result file")
			with open(result_file_name, "a") as f:
				f.write("%d,%d,%s,%s,%f,%f" % (no_of_layers, no_of_neurons, activation_name, model_checkpoint_file, model_avg_error, simulation_model_avg_error))
				f.write("\n")
				f.close()
			print("Removing the job file %s" % job)
			os.remove(job)