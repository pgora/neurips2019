# This file contains utils for preparing data for training

import numpy as np
from sklearn.preprocessing import StandardScaler 

# Scales traffic light settings to [-1.0, 1.0]. Also includes some other methods useful when 
# going from the original traffic light setting space to [-1.0, 1.0] and back to the original
class TrafficLightsScaler:
	# function that transforms inputs in [0, 119] to inputs in [-1.0, 1.0], using simple linear scaling
	def transform(self, X):
		return 2.0 * (X/120.0 - 0.5)
	# function that scales inputs in [-1.0, 1.0] back to inputs in [0, 119], using simple linear scaling
	# no rounding is applied here (cf. 'get_original_inputs' fuction)
	def inverse_transform(self, X_transformed):
		return (X_transformed / 2.0 + 0.5) * 120.0
	# function for rounding [-1.0,1.0] inputs so that they correspond to integer entries in the original traffic light setting space
	def round_tanh_inputs(self, inpt):
		return 2.0 * (np.round(120.0 * (inpt / 2.0 + 0.5))/120.0 - 0.5)    
	# function that returns inputs in a form accepted by simulator, given inputs in [-1.0,1.0]. Rounding is applied to get integers
	def get_original_inputs(self, tanh_inputs):
		return (np.round(120.0 * (tanh_inputs / 2.0 + 0.5)) % 120.0).astype(int)

# Scales standard traffic lights data (as in 100k dataset)
# Returns normalized data (X and y separately) and the 
# corresponding scalers
def scale_standard_traffic_light_data(traffic_lights_data):
	no_of_columns = traffic_lights_data.shape[1]
	traffic_lights_normalized = traffic_lights_data.copy()
	X_scaler = TrafficLightsScaler()
	traffic_lights_normalized.iloc[:,0:(no_of_columns-1)] = X_scaler.transform(traffic_lights_normalized.iloc[:,0:(no_of_columns-1)])
	y_scaler = StandardScaler()
	y_scaler.fit(traffic_lights_normalized.iloc[:,no_of_columns-1].values.reshape(-1,1))
	traffic_lights_normalized.iloc[:,no_of_columns-1] = y_scaler.transform(traffic_lights_normalized.iloc[:,no_of_columns-1].values.reshape(-1,1))
	X = traffic_lights_normalized.iloc[:,0:(no_of_columns-1)]
	y = traffic_lights_normalized.iloc[:,no_of_columns-1]
	return X, y, X_scaler, y_scaler

# Iterates over a dataset in batches.
# There is an option to permute the dataset
# after each epoch, set to False by default
class BatchIterator:
    def __init__(self, X, y, batch_size, permute_data_after_each_epoch=False):
        assert X.shape[0] == y.shape[0]
        self.X = X.copy()
        self.y = y.copy()
        self.n = X.shape[0]
        self.batch_size = batch_size
        self.batch_start = 0
        self.permute_data_after_each_epoch = permute_data_after_each_epoch
    def next(self):
        max_index = min(self.batch_start + self.batch_size, self.n)
        indices_to_select = list(range(self.batch_start, max_index))
        self.batch_start += self.batch_size
        if self.batch_start >= self.n and self.permute_data_after_each_epoch:
        	data_permutation = np.random.permutation(self.n)
        	self.X = self.X[data_permutation]
        	self.y = self.y[data_permutation]
        self.batch_start %= self.n
        X_batch = self.X.iloc[indices_to_select,:]
        y_batch = self.y.iloc[indices_to_select]      
        return X_batch, y_batch