# This file contains methods used for building graph neural networks

import numpy as np
import tensorflow as tf

from graph_utils import extract_undirected_edges

sqrt_of_two = np.sqrt(2.0)

# Constructs 'transfer matrix' from undirected edge list. Columns of this matrix
# correspond to (undirected) edges. Nonzero entries in each column correspond
# to the ends of the respective edge. They are randomly initialized using
# Glorot initialization with normal variables. 
# The list of undirected edges given to the method is assumed to be ordered
# using lexicographical order
# The matrix may be returned in transpose form if required
def get_transfer_matrix_plus_bias_vector(undirected_edges, transpose):
    rowno = max(np.array(undirected_edges).flatten()) + 1
    colno = len(undirected_edges)
    params = tf.eye(colno)
    coords_ids_weights = []
    i = 0
    if transpose:
        column_nonzero_counts = np.zeros([rowno])
    for e in undirected_edges:
        coords_ids_weights.append([[e[0],i],i,1.0])
        coords_ids_weights.append([[e[1],i],i,1.0])
        if transpose:
            column_nonzero_counts[e[0]] += 1.0
            column_nonzero_counts[e[1]] += 1.0
        i += 1
    coords_ids_weights = sorted(coords_ids_weights,key=lambda x: colno * x[0][0] + x[0][1])
    coords = [x[0] for x in coords_ids_weights]
    ids = [x[1] for x in coords_ids_weights]
    weights = tf.Variable(tf.random_normal([len(coords_ids_weights)]))
    # can be used for testing if factors for Glorot initialization are calculated correctly
    #weights = tf.ones([len(coords_ids_weights)])
    sp_ids = tf.SparseTensor(dense_shape=[rowno, colno],indices=coords,values=ids)
    sp_weights = tf.SparseTensor(dense_shape=[rowno, colno],indices=coords,values=weights)
    edge_matrix = tf.nn.embedding_lookup_sparse(params=params,sp_ids=sp_ids,sp_weights=sp_weights,combiner="sum")
    # this seems necessary to get the shapes right, though conceptually it changes nothing
    edge_matrix = tf.reshape(edge_matrix, [rowno, colno])
    # transposition (optional) and Glorot initialization
    if transpose:
        edge_matrix = tf.transpose(edge_matrix)
        # in random graph experiments, there may be disconnected vertices
        glorot_weights = np.maximum(np.sqrt(column_nonzero_counts), 1)
        edge_matrix /= glorot_weights
        biases = tf.Variable(tf.zeros([rowno]))
    else:
        glorot_weights = sqrt_of_two * np.ones([colno])
        edge_matrix /= glorot_weights
        biases = tf.Variable(tf.zeros([colno]))
    return edge_matrix, biases

# Constructs 'transfer matrix' a la Kipf (https://arxiv.org/abs/1609.02907) from undirected edge list.
# Columns and rows of this matrix correspond to vertices of the adjacency graph. Nonzero entries in 
# each column correspond to vertices adjacent to a given vertex. They are randomly initialized using
# Glorot initialization with normal variables. The list of undirected edges given to the method is 
# assumed to be ordered using lexicographical order.
def get_kipfs_transfer_matrix_plus_bias_vector(undirected_edges):
    rowno = max(np.array(undirected_edges).flatten()) + 1
    params = tf.eye(rowno)
    coords_ids_weights = []
    column_nonzero_counts = np.zeros([rowno])
    for e in undirected_edges:
        coords_ids_weights.append([[e[0],e[1]],e[1],1.0])
        coords_ids_weights.append([[e[1],e[0]],e[0],1.0])
        column_nonzero_counts[e[0]] += 1.0
        column_nonzero_counts[e[1]] += 1.0
    for i in range(rowno):
    	coords_ids_weights.append([[i,i],i,1.0])
    	column_nonzero_counts[i] += 1.0
    coords_ids_weights = sorted(coords_ids_weights,key=lambda x: rowno * x[0][0] + x[0][1])
    coords = [x[0] for x in coords_ids_weights]
    ids = [x[1] for x in coords_ids_weights]
    weights = tf.Variable(tf.random_normal([len(coords_ids_weights)]))
    # can be used for testing if factors for Glorot initialization are calculated correctly:
    # weights = tf.ones([len(coords_ids_weights)])
    sp_ids = tf.SparseTensor(dense_shape=[rowno, rowno],indices=coords,values=ids)
    sp_weights = tf.SparseTensor(dense_shape=[rowno, rowno],indices=coords,values=weights)
    edge_matrix = tf.nn.embedding_lookup_sparse(params=params,sp_ids=sp_ids,sp_weights=sp_weights,combiner="sum")
    # this seems necessary to get the shapes right, though conceptually it changes nothing
    edge_matrix = tf.reshape(edge_matrix, [rowno, rowno])
    
    glorot_weights = np.maximum(np.sqrt(column_nonzero_counts), 1)
    edge_matrix /= glorot_weights
    biases = tf.Variable(tf.zeros([rowno]))
    
    return edge_matrix, biases

# Like 'get_transfer_matrix_plus_bias_vector', constructs a 'transfer matrix'.
# However, this time weights in the columns of the matrix are shared between
# the columns. This only seems to make sense for non-transposed transfer
# matrices, hence no option to transpose the matrix is included. 
# There is an option to share biases as well, set to False by default
def get_shared_transfer_matrix_plus_bias_vector(undirected_edges, share_biases=False):
    rowno = max(np.array(undirected_edges).flatten()) + 1
    colno = len(undirected_edges)
    params = tf.eye(colno)
    coords_ids_weights = []
    i = 0
    for e in undirected_edges:
        coords_ids_weights.append([[e[0],i],i,1.0])
        coords_ids_weights.append([[e[1],i],i,1.0])
        i += 1
    indices = list(range(len(coords_ids_weights)))
    coords_ids_weights_with_indices = list(zip(coords_ids_weights, indices))
    coords_ids_weights_with_indices = sorted(coords_ids_weights_with_indices,key=lambda x: colno * x[0][0][0] + x[0][0][1])
    coords_ids_weights = [x[0] for x in coords_ids_weights_with_indices]
    indices = [x[1] for x in coords_ids_weights_with_indices]
    coords = [x[0] for x in coords_ids_weights]
    ids = [x[1] for x in coords_ids_weights]
    weights = tf.Variable(tf.random_normal([2]))
    weights = tf.tile(weights, [len(coords_ids_weights)])
    weights = tf.gather(weights, indices)
    # can be used for testing if factors for Glorot initialization are calculated correctly
    #weights = tf.ones([len(coords_ids_weights)])
    sp_ids = tf.SparseTensor(dense_shape=[rowno, colno],indices=coords,values=ids)
    sp_weights = tf.SparseTensor(dense_shape=[rowno, colno],indices=coords,values=weights)
    edge_matrix = tf.nn.embedding_lookup_sparse(params=params,sp_ids=sp_ids,sp_weights=sp_weights,combiner="sum")
    # this seems necessary to get the shapes right, though conceptually it changes nothing
    edge_matrix = tf.reshape(edge_matrix, [rowno, colno])
    # Glorot initialization
    glorot_weights = sqrt_of_two * np.ones([colno])
    edge_matrix /= glorot_weights
    if share_biases:
        biases = tf.Variable(tf.zeros([1]))
        biases = tf.tile(biases, [colno])
    else:
        biases = tf.Variable(tf.zeros([colno]))
    return edge_matrix, biases

# Construct a graph neural net using transfer matrices (alternatingly plain and transposed)
# One can specify the number of layers and the number of 'colour' channels, assumed to be 
# equal troughout the network. Activation function and (undirected) adjacency matrix need
# to be provided. An option is provided to share weights in (non-transposed) transfer matrices,
# set to false by default.
# Input to the net should be a two- or a three-dimensional tensor, first dimension corresponding
# to batch size, second dimension to graph vertex localization, and the third (if present)
# to the number of channels.
def transfer_matrix_neural_net(inpt, 
	no_of_layers, no_of_channels, activation_function, adjacency_matrix, 
	share_weights_in_transfer_matrix=False, share_biases_in_transfer_matrix=False,
	verbose=False):
	assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1]
	assert len(inpt.shape) == 2 or len(inpt.shape) == 3
	edges = extract_undirected_edges(adjacency_matrix)
	if len(inpt.shape) == 3:
		hidden = inpt		
		no_in_channels = int(inpt.shape[2])
	else:
		hidden = tf.expand_dims(inpt,axis=2)
		no_in_channels = 1
	no_out_channels = no_of_channels
	for i in range(no_of_layers):
		transfer_matrix_stacks = []
		bias_vector_stacks = []
		for k in range(no_out_channels):
			transfer_matrices = []
			for c in range(no_in_channels):
				if i % 2 == 0:
					if share_weights_in_transfer_matrix:
						transfer_matrix, bias = get_shared_transfer_matrix_plus_bias_vector(edges, share_biases_in_transfer_matrix)
					else:
						transfer_matrix, bias = get_transfer_matrix_plus_bias_vector(edges, False)
				else:
					transfer_matrix, bias = get_transfer_matrix_plus_bias_vector(edges, True)
				transfer_matrices.append(transfer_matrix)
			transfer_matrix_stack = tf.stack(transfer_matrices,axis=-1)
			transfer_matrix_stacks.append(transfer_matrix_stack)
			bias_vector_stacks.append(bias)
		transfer_matrix = tf.stack(transfer_matrix_stacks,axis=-1)
		bias = tf.stack(bias_vector_stacks,axis=-1)
		if verbose:
			print(transfer_matrix.shape)
		hidden = tf.tensordot(hidden,transfer_matrix,[[1,2],[0,2]])
        # A sqrt factor used to account for multiple channels (like in Glorot initialization)
		hidden /= np.sqrt(no_of_channels)
		hidden = hidden + bias
		hidden = activation_function(hidden)
		no_in_channels = no_of_channels
		if verbose:
			print(hidden.shape)
	hidden = tf.contrib.layers.flatten(hidden)
	if verbose:
		print(hidden.shape)
	return tf.layers.Dense(units=1,activation=tf.identity)(hidden)

# Construct a graph neural net using transfer matrices (alternatingly plain and transposed)
# One can specify the number of layers and the number of 'colour' channels for each layer.
# Activation function and (undirected) adjacency matrix need to be provided.
# Unlike 'transfer_matrix_neural_net', this version accepts a list of channel numbers for 
# the consecutive network layers.
# Input to the net should be a two- or a three-dimensional tensor, first dimension corresponding
# to batch size, second dimension to graph vertex localization, and the third (if present)
# to the number of channels.
def transfer_matrix_neural_net_var_channels(inpt, 
	no_of_layers, no_of_channels, activation_function, adjacency_matrix, 
	share_weights_in_transfer_matrix=False, share_biases_in_transfer_matrix=False,
	verbose=False):
	assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1]
	assert len(no_of_channels) == no_of_layers
	assert len(inpt.shape) == 2 or len(inpt.shape) == 3
	edges = extract_undirected_edges(adjacency_matrix)
	if len(inpt.shape) == 3:
		hidden = inpt		
		no_in_channels = int(inpt.shape[2])
	else:
		hidden = tf.expand_dims(inpt,axis=2)
		no_in_channels = 1
	for i in range(no_of_layers):
		no_out_channels = no_of_channels[i]
		transfer_matrix_stacks = []
		bias_vector_stacks = []
		for k in range(no_out_channels):
			transfer_matrices = []
			for c in range(no_in_channels):
				if i % 2 == 0:
					if share_weights_in_transfer_matrix:
						transfer_matrix, bias = get_shared_transfer_matrix_plus_bias_vector(edges, share_biases_in_transfer_matrix)
					else:
						transfer_matrix, bias = get_transfer_matrix_plus_bias_vector(edges, False)
				else:
						transfer_matrix, bias = get_transfer_matrix_plus_bias_vector(edges, True)
				transfer_matrices.append(transfer_matrix)
			transfer_matrix_stack = tf.stack(transfer_matrices,axis=-1)
			transfer_matrix_stacks.append(transfer_matrix_stack)
			bias_vector_stacks.append(bias)
		transfer_matrix = tf.stack(transfer_matrix_stacks,axis=-1)
		bias = tf.stack(bias_vector_stacks,axis=-1)
		if verbose:
			print(transfer_matrix.shape)
		hidden = tf.tensordot(hidden,transfer_matrix,[[1,2],[0,2]])
		# A sqrt factor used to account for multiple channels (like in Glorot initialization)
		hidden /= np.sqrt((no_in_channels + no_out_channels)/2.0)
		hidden = hidden + bias
		hidden = activation_function(hidden)
		no_in_channels = no_out_channels
		if verbose:
			print(hidden.shape)
	hidden = tf.contrib.layers.flatten(hidden)
	if verbose:
		print(hidden.shape)
		return tf.layers.Dense(units=1,activation=tf.identity)(hidden)

# Construct a graph neural net using transfer matrices (alternatingly plain and transposed)
# One can specify the number of layers and the number of 'colour' channels, assumed to be 
# equal troughout the network. Activation function and (undirected) adjacency matrix need
# to be provided. An option is provided to share weights in (non-transposed) transfer matrices,
# set to false by default.
# Input to the net should be a two- or a three-dimensional tensor, first dimension corresponding
# to batch size, second dimension to graph vertex localization, and the third (if present)
# to the number of channels.
# In addition to the final layer output tensor, the intermediate layer list is returned
# as well (as the second output, a list which also includes the final layer as its last
# element)
def transfer_matrix_neural_net_with_layer_output(inpt, 
	no_of_layers, no_of_channels, activation_function, adjacency_matrix, 
	share_weights_in_transfer_matrix=False, share_biases_in_transfer_matrix=False,
	verbose=False):
	assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1]
	assert len(inpt.shape) == 2 or len(inpt.shape) == 3
	edges = extract_undirected_edges(adjacency_matrix)
	layers = [inpt]
	if len(inpt.shape) == 3:
		hidden = inpt		
		no_in_channels = int(inpt.shape[2])
	else:
		hidden = tf.expand_dims(inpt,axis=2)
		no_in_channels = 1
	no_out_channels = no_of_channels
	for i in range(no_of_layers):
		transfer_matrix_stacks = []
		bias_vector_stacks = []
		for k in range(no_out_channels):
			transfer_matrices = []
			for c in range(no_in_channels):
				if i % 2 == 0:
					if share_weights_in_transfer_matrix:
						transfer_matrix, bias = get_shared_transfer_matrix_plus_bias_vector(edges, share_biases_in_transfer_matrix)
					else:
						transfer_matrix, bias = get_transfer_matrix_plus_bias_vector(edges, False)
				else:
					transfer_matrix, bias = get_transfer_matrix_plus_bias_vector(edges, True)
				transfer_matrices.append(transfer_matrix)
			transfer_matrix_stack = tf.stack(transfer_matrices,axis=-1)
			transfer_matrix_stacks.append(transfer_matrix_stack)
			bias_vector_stacks.append(bias)
		transfer_matrix = tf.stack(transfer_matrix_stacks,axis=-1)
		bias = tf.stack(bias_vector_stacks,axis=-1)
		if verbose:
			print(transfer_matrix.shape)
		hidden = tf.tensordot(hidden,transfer_matrix,[[1,2],[0,2]])
        # A sqrt factor used to account for multiple channels (like in Glorot initialization)
		hidden /= np.sqrt(no_of_channels)
		hidden = hidden + bias
		hidden = activation_function(hidden)
		layers.append(hidden)
		no_in_channels = no_of_channels
		if verbose:
			print(hidden.shape)
	hidden = tf.contrib.layers.flatten(hidden)
	if verbose:
		print(hidden.shape)
	out_layer = tf.layers.Dense(units=1,activation=tf.identity)(hidden)
	layers.append(out_layer)
	return out_layer, layers

# Construct a graph neural net using transfer matrices a la Kipf (https://arxiv.org/abs/1609.02907)
# One can specify the number of layers and the number of 'colour' channels, assumed to be 
# equal troughout the network. Activation function and (undirected) adjacency matrix need
# to be provided.
# Input to the net should be a two- or a three-dimensional tensor, first dimension corresponding
# to batch size, second dimension to graph vertex localization, and the third (if present)
# to the number of channels.
def kipfs_transfer_matrix_neural_net(inpt, 
	no_of_layers, no_of_channels, activation_function, adjacency_matrix, 
	verbose=False):
	assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1]
	assert len(inpt.shape) == 2 or len(inpt.shape) == 3
	edges = extract_undirected_edges(adjacency_matrix)
	if len(inpt.shape) == 3:
		hidden = inpt		
		no_in_channels = int(inpt.shape[2])
	else:
		hidden = tf.expand_dims(inpt,axis=2)
		no_in_channels = 1
	no_out_channels = no_of_channels
	for i in range(no_of_layers):
		transfer_matrix_stacks = []
		bias_vector_stacks = []
		for k in range(no_out_channels):
			transfer_matrices = []
			for c in range(no_in_channels):
				transfer_matrix, bias = get_kipfs_transfer_matrix_plus_bias_vector(edges)
				transfer_matrices.append(transfer_matrix)
			transfer_matrix_stack = tf.stack(transfer_matrices,axis=-1)
			transfer_matrix_stacks.append(transfer_matrix_stack)
			bias_vector_stacks.append(bias)
		transfer_matrix = tf.stack(transfer_matrix_stacks,axis=-1)
		bias = tf.stack(bias_vector_stacks,axis=-1)
		if verbose:
			print(transfer_matrix.shape)
		hidden = tf.tensordot(hidden,transfer_matrix,[[1,2],[0,2]])
        # A sqrt factor used to account for multiple channels (like in Glorot initialization)
		hidden /= np.sqrt(no_of_channels)
		hidden = hidden + bias
		hidden = activation_function(hidden)
		no_in_channels = no_of_channels
		if verbose:
			print(hidden.shape)
	hidden = tf.contrib.layers.flatten(hidden)
	if verbose:
		print(hidden.shape)
	return tf.layers.Dense(units=1,activation=tf.identity)(hidden)

# Construct a graph neural net using transfer matrices a la Kipf (https://arxiv.org/abs/1609.02907)
# One can specify the number of layers and the number of 'colour' channels, assumed to be 
# equal troughout the network. Activation function and (undirected) adjacency matrix need
# to be provided.
# Input to the net should be a two- or a three-dimensional tensor, first dimension corresponding
# to batch size, second dimension to graph vertex localization, and the third (if present)
# to the number of channels.
# In addition to the final layer output tensor, the intermediate layer list is returned
# as well (as the second output, a list which also includes the final layer as its last
# element)
def kipfs_transfer_matrix_neural_net_with_layer_output(inpt, 
	no_of_layers, no_of_channels, activation_function, adjacency_matrix, 
	verbose=False):
	assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1]
	assert len(inpt.shape) == 2 or len(inpt.shape) == 3
	edges = extract_undirected_edges(adjacency_matrix)
	layers = [inpt]
	if len(inpt.shape) == 3:
		hidden = inpt		
		no_in_channels = int(inpt.shape[2])
	else:
		hidden = tf.expand_dims(inpt,axis=2)
		no_in_channels = 1
	no_out_channels = no_of_channels
	for i in range(no_of_layers):
		transfer_matrix_stacks = []
		bias_vector_stacks = []
		for k in range(no_out_channels):
			transfer_matrices = []
			for c in range(no_in_channels):
				transfer_matrix, bias = get_kipfs_transfer_matrix_plus_bias_vector(edges)
				transfer_matrices.append(transfer_matrix)
			transfer_matrix_stack = tf.stack(transfer_matrices,axis=-1)
			transfer_matrix_stacks.append(transfer_matrix_stack)
			bias_vector_stacks.append(bias)
		transfer_matrix = tf.stack(transfer_matrix_stacks,axis=-1)
		bias = tf.stack(bias_vector_stacks,axis=-1)
		if verbose:
			print(transfer_matrix.shape)
		hidden = tf.tensordot(hidden,transfer_matrix,[[1,2],[0,2]])
        # A sqrt factor used to account for multiple channels (like in Glorot initialization)
		hidden /= np.sqrt(no_of_channels)
		hidden = hidden + bias
		hidden = activation_function(hidden)
		layers.append(hidden)
		no_in_channels = no_of_channels
		if verbose:
			print(hidden.shape)
	hidden = tf.contrib.layers.flatten(hidden)
	if verbose:
		print(hidden.shape)
	out_layer = tf.layers.Dense(units=1,activation=tf.identity)(hidden)
	layers.append(out_layer)
	return out_layer, layers
