# This file contains methods for neural net training and evaluation

import numpy as np
import tensorflow as tf

from time import sleep
from random import randint

import requests
import json
import queue
import threading

import data_preparation_utils
import iterative_updaters

# Function for training and saving neural net models. Needs neural net output tensor,
# input tensor and target tensor, as well as a batch iterator and and model
# checkpoint file path. Returns a trajectory of test and batch losses
def train_model(neural_net_output_tensor, input_tensor, target_tensor,
	optimizer, no_of_iterations, batch_iterator, X_test_data, y_test_data, model_checkpoint_file_path, 
	checkpoint_save_frequency, verbose=True):

	loss = tf.losses.mean_squared_error(labels=target_tensor,predictions=neural_net_output_tensor)
	minimize_loss = optimizer.minimize(loss)

	sess = tf.Session()
	saver = tf.train.Saver()
	init = tf.global_variables_initializer()
	sess.run(init)

	test_and_batch_losses = []

	for i in range(no_of_iterations):
		X_batch, y_batch = batch_iterator.next()
		y_batch = y_batch.values.reshape([-1,1])
		l, _ = sess.run([loss, minimize_loss], feed_dict={input_tensor: X_batch, target_tensor: y_batch})
		if i % checkpoint_save_frequency == 0:
			saver.save(sess, model_checkpoint_file_path)
			tl = sess.run(loss, feed_dict={input_tensor: X_test_data, target_tensor: y_test_data.values.reshape([-1,1])})
			test_and_batch_losses.append((tl, l))
			if verbose:
				print("Test loss: %f, batch loss: %f, model saved under %s" % (tl, l, model_checkpoint_file_path))

	sess.close()
	# return the trajectory of test and batch losses, recorded when saving the checkpoints
	return test_and_batch_losses

# Evaluates a saved model on a dataset. Normalized X and y for the dataset, as well as 
# a y scaler for calculating inverse transform, need to be provided
def evaluate_model_on_a_dataset(model_checkpoint_file_path, model_output_tensor, model_input_tensor, X, y, y_scaler):
	sess =  tf.Session()
	saver = tf.train.Saver()
	saver.restore(sess, model_checkpoint_file_path)

	out = y_scaler.inverse_transform(sess.run(model_output_tensor, feed_dict={model_input_tensor:X}).reshape(-1))
	target = y_scaler.inverse_transform(y)
	model_avg_error = np.mean(abs(out-target)/target)

	sess.close()

	sorting_permutation = np.argsort(-y)
	y_test_sorted = y_scaler.inverse_transform(y)[sorting_permutation]
	y_test_pred_sorted = out[sorting_permutation]
	actual_vs_predicted = list(zip(y_test_sorted, y_test_pred_sorted))

	return model_avg_error, actual_vs_predicted

# Evaluates a saved model on a dataset using maximum relative error as a metrics. Normalized X and y for the dataset,
# as well as a y scaler for calculating inverse transform, need to be provided
def find_model_maximum_relative_error_on_a_dataset(model_checkpoint_file_path, model_output_tensor, model_input_tensor, X, y, y_scaler):
	sess =  tf.Session()
	saver = tf.train.Saver()
	saver.restore(sess, model_checkpoint_file_path)

	out = y_scaler.inverse_transform(sess.run(model_output_tensor, feed_dict={model_input_tensor:X}).reshape(-1))
	target = y_scaler.inverse_transform(y)
	model_max_error = np.max(abs(out-target)/target)

	sess.close()

	return model_max_error

# Evaluates a saved model on a dataset using relative error standard deviation as a metrics. 
# Normalized X and y for the dataset, as well as a y scaler for calculating inverse transform, need to be provided
def find_model_relative_error_stdev_on_a_dataset(model_checkpoint_file_path, model_output_tensor, model_input_tensor, X, y, y_scaler):
	sess =  tf.Session()
	saver = tf.train.Saver()
	saver.restore(sess, model_checkpoint_file_path)

	out = y_scaler.inverse_transform(sess.run(model_output_tensor, feed_dict={model_input_tensor:X}).reshape(-1))
	target = y_scaler.inverse_transform(y)
	model_max_error = np.std(abs(out-target)/target)

	sess.close()

	return model_max_error

# Evaluates a saved toy model on a dataset. Normalized X and y for the dataset, as well as 
# a y scaler for calculating inverse transform, need to be provided
def evaluate_toy_model_on_a_dataset(model_checkpoint_file_path, model_output_tensor, model_input_tensor, X, y, y_scaler):
	sess =  tf.Session()
	saver = tf.train.Saver()
	saver.restore(sess, model_checkpoint_file_path)

	out0 = sess.run(model_output_tensor, feed_dict={model_input_tensor:X}).reshape(-1)
	out = y_scaler.inverse_transform(out0)
	target = y_scaler.inverse_transform(y)
	model_mse = np.mean((out0 - y)*(out0 - y))

	sess.close()

	sorting_permutation = np.argsort(-y)
	y_test_sorted = y_scaler.inverse_transform(y)[sorting_permutation]
	y_test_pred_sorted = out[sorting_permutation]
	actual_vs_predicted = list(zip(y_test_sorted, y_test_pred_sorted))

	return model_mse, actual_vs_predicted

# Gets gradient function for a given neural net. Input tensor, as well as
# input scaler need to be specified. The inputs to the function returned
# are not constrained, however, they are transformed by tanh before
# passing them to the neural net. This way, inputs to the neural net
# are always in [-1.0,1.0]
def get_gradient_function(sess, neural_net_output_tensor, input_tensor, X_scaler):
	grad_op = tf.gradients(neural_net_output_tensor, input_tensor)
	# Calculates the gradient of a neural net output with respect
	# to unconstrained inputs that are transformed by tanh before
	# actually feeding them to the net
	def evaluate_gradient(input_guess_before_tanh):
		input_tanh = np.tanh(input_guess_before_tanh)
		rounded_input_tanh = X_scaler.round_tanh_inputs(input_tanh)
		tanh_grad = sess.run(grad_op, feed_dict={input_tensor: input_tanh})
		grad = tanh_grad * (1.0 - input_tanh * input_tanh)
		return grad[0]
	return lambda x: evaluate_gradient(x)

# Generates a gradient descent trajectory for a given neural net model.
# The neural net output and input tensors need to be provided, as well as
# X and y data scalers, an iterative updater of user's choice, 
# the total number of steps in gradient descent and trajectory probing frequency
# (for use when saving trajectory points. If set to 1, all steps will be saved) 
def generate_gradient_descent_trajectory(sess, neural_net_output_tensor, input_tensor, 
	X_scaler, y_scaler, updater, total_no_of_steps, probing_frequency, verbose=False):
	
	gradient = get_gradient_function(sess, neural_net_output_tensor, input_tensor, X_scaler)
	
	# initial guess to be fed into tanh
	input_guess = np.random.uniform(high=0.7,low=-0.7,size=[1,input_tensor.shape[1]])

	# (input, predicted value) trajectory from gradient optimization
	trajectory = []

	for i in range(total_no_of_steps):
	    if i % probing_frequency == 0:
	    	input_tanh = np.tanh(input_guess)
	    	rounded_input_tanh = X_scaler.round_tanh_inputs(input_tanh)
	    	original_inputs = X_scaler.get_original_inputs(rounded_input_tanh)
	    	pr = sess.run(neural_net_output_tensor, feed_dict={input_tensor: rounded_input_tanh})
	    	pr = y_scaler.inverse_transform(pr)
	    	trajectory.append((original_inputs.tolist()[0], pr[0,0]))
	    	if verbose:
	    		p = sess.run(neural_net_output_tensor, feed_dict={input_tensor: input_tanh})
	    		p = y_scaler.inverse_transform(p)
	    		print("Prediction %f, rounded: %f" % (p, pr))
	    input_guess = updater.update(input_guess, gradient)

	return trajectory

# Generates multiple gradient descent trajectories and joins them together
def generate_and_join_multiple_gradient_descent_trajectories(sess, no_of_trajectories, neural_net_output_tensor, input_tensor, 
	X_scaler, y_scaler, updater, total_no_of_steps, probing_frequency, verbose=True, trajectories_verbose=False):

	trajectories = []

	for i in range(no_of_trajectories):
		if verbose:
			print("Generating trajectory %d" % (i+1))
		trajectory = generate_gradient_descent_trajectory(sess, neural_net_output_tensor, input_tensor, X_scaler, y_scaler, updater, total_no_of_steps, probing_frequency, verbose=trajectories_verbose)
		trajectories += trajectory

	return trajectories

# Generates a test set for neural net evaluation 
def generate_test_set_from_trajectory_points(trajectory,
	no_of_threads, microservice_address='http://18.195.81.162:25041/',
	verbose=True):
	if verbose:
		print("Creating queues")

	input_queue = queue.Queue(len(trajectory))
	output_queue = queue.Queue(len(trajectory))

	if verbose:
		print("Getting locks")

	input_queue_lock = threading.Lock()
	output_queue_lock = threading.Lock()

	if verbose:
		print("Filling the queue")                

	input_queue_lock.acquire()
	for p in trajectory:
		input_queue.put(p)
	input_queue_lock.release()

	class simulationRequestThread(threading.Thread):
		def __init__(self, threadID, name, input_queue, output_queue, input_queue_lock, output_queue_lock):
			threading.Thread.__init__(self)
			self.threadID = threadID
			self.name = name
			self.input_queue = input_queue
			self.output_queue = output_queue
			self.input_queue_lock = input_queue_lock
			self.output_queue_lock = output_queue_lock
		def run(self):
			exit_flag = False
			while not exit_flag:
				self.input_queue_lock.acquire()
				if not self.input_queue.empty():
					point_to_process, estimated_waiting_time = self.input_queue.get()
					self.input_queue_lock.release()
					if verbose:
						print("Processing point: %s" % str(point_to_process))
					sleep(randint(10,100)/100.0)
					try:
						resp = requests.post(microservice_address, json={'user':'pgora', 'password':'Ghzf8ftb', 'settings':str(point_to_process)}, headers={'Content-Type': 'application/json'})
						self.output_queue_lock.acquire()
						self.output_queue.put((point_to_process,float(json.loads(resp.text)['score'])))
						print(float(json.loads(resp.text)['score']))
						# an earlier version where waiting times were directly compared - kept as a backup
						#self.output_queue.put((estimated_waiting_time, float(json.loads(resp.text)['score'])))
						self.output_queue_lock.release()
					except:
						pass
				else:
					self.input_queue_lock.release()
					exit_flag = True

    # create and start the threads
	threads = []
	threadList = ["T%d" % d for d in range(no_of_threads)]

	threadID = 1

	if verbose:
		print("Starting the threads")

	for tName in threadList:
		thread = simulationRequestThread(threadID, tName, input_queue, output_queue, input_queue_lock, output_queue_lock)
		thread.start()
		threads.append(thread)
		threadID += 1

	# wait for the threads to complete
	for t in threads:
		t.join()

	Xes = []
	ys = []
    
	# extract the Xes and ys for the test set
	output_queue_lock.acquire()
	while not output_queue.empty():
		res = output_queue.get()
		Xes.append(res[0])
		ys.append(res[1])
		if verbose:
			print(res)
	output_queue_lock.release()

	# convert to numpy
	Xes = np.array(Xes)
	ys = np.array(ys)

	return Xes, ys