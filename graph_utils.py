# This file contains several utility methods for working with graphs and their adjacency matrices

import random
import numpy as np

# A basic utility method for graph neural networks. 
# Extracts undirected edges from a (directed or undirected) adjacency matrix.
# Whenever there is a link between two vertices in one direction or the other,
# an edge will be returned. Edges are represented by pairs (i, j) such that
# i < j. The returned list is lexicographically ordered
def extract_undirected_edges(adj_matrix):
    # adjacency matrix should be square
    assert adj_matrix.shape[0] == adj_matrix.shape[1]
    n = adj_matrix.shape[0]
    edges = set()
    for i in range(n):
        for j in range(n):
            if adj_matrix[i, j] == 1:
                if i < j:
                    edges.add((i, j))
                elif j < i:
                    edges.add((j, i))
    edges = sorted(list(edges), key=lambda x: n * x[0] + x[1])
    return edges

# Generates a random adjacency matrix with the required number of graph nodes
# and edges. By convention, nonzero entries are only generated below diagonal,
# and it is assumed that it will be passed to an undirected edges extractor
# anyway
def generate_random_adjacency_matrix(number_of_vertices, number_of_edges):
	possible_edges = [(i, j) for i in range(number_of_vertices) for j in range(number_of_vertices) if i < j]
	edges_to_keep = random.sample(possible_edges, number_of_edges)
	adjacency_matrix = [[int((i,j) in edges_to_keep or i == j) for i in range(number_of_vertices)] for j in range(number_of_vertices)]
	return np.array(adjacency_matrix)

# Compares two adjacency matrices by calculating the number of elements in the 
# symmetric difference of their edge set. The matrices are symmetrized before
# the calculation, so that the information about the direction of the edges,
# if present, is lost.
def undirected_symmetric_difference(adjacency_matrix_1, adjacency_matrix_2):
	undirected_edges_1 = set(extract_undirected_edges(adjacency_matrix_1))
	undirected_edges_2 = set(extract_undirected_edges(adjacency_matrix_2))
	symmetric_diff = (undirected_edges_1 - undirected_edges_2) | (undirected_edges_2 - undirected_edges_1)
	return len(symmetric_diff)

# Generates a random adjacency matrix with a fixed symmetric difference with respect to 
# a given one, keeping the number of edges equal. The corresponding graph is required
# to be connected by default, though there is an option to change this behaviour.
# By convention, nonzero entries are only generated below diagonal, and it is assumed 
# that the output of this function will be passed to an undirected edges extractor,
# which makes this choice irrelevant
def generate_random_adjacency_matrix_with_fixed_symmetric_difference(symetric_difference_div_by_two, 
	reference_adjacency_matrix, generate_connected_graph = True):
	assert len(reference_adjacency_matrix.shape) == 2
	assert reference_adjacency_matrix.shape[0] == reference_adjacency_matrix.shape[1]
	n = reference_adjacency_matrix.shape[0]
	all_undirected_edges = set([(i, j) for i in range(n) for j in range(n) if i < j])
	reference_undirected_edges = extract_undirected_edges(reference_adjacency_matrix)
	adjacency_matrix = []
	while len(adjacency_matrix) == 0:
		reference_non_edges = list(all_undirected_edges - set(reference_undirected_edges))
		edges_to_remove = random.sample(reference_undirected_edges, symetric_difference_div_by_two)
		edges_to_add = random.sample(reference_non_edges, symetric_difference_div_by_two)
		edges = (set(reference_undirected_edges) - set(edges_to_remove)) | set(edges_to_add)
		vertices = set()
		for (i, j) in edges:
			if i != j:
				vertices.add(i)
				vertices.add(j)
		print("%d %d" % (len(vertices), n))
		if len(vertices) == n:
			adjacency_matrix = [[int((i,j) in edges or i == j) for i in range(n)] for j in range(n)]
	return np.array(adjacency_matrix)

# Generates a random permutation of n elements which has approximately k transpositions
# n -> no_of_elements, k -> no_of_transpositions
def generate_random_permutation_with_approx_no_of_transpositions(no_of_elements, no_of_transpositions):
	permutation = list(range(no_of_elements))
	for l in range(no_of_transpositions):
		i = random.randint(0, no_of_elements-1)
		j = random.randint(0, no_of_elements-1)
		value_at_i = permutation[i]
		permutation[i] = permutation[j]
		permutation[j] = value_at_i
	return permutation

