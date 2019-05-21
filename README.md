# Graph based sparse neural networks for traffic signal optimization

## Summary

This repository contains companion code for the paper *Graph based sparse neural networks for traffic signal optimization*, submitted for NeurIPS Conference 2019.

Trained models from our experiments can be found in the file ``100k_fit_and_eval_experiment_results.zip`` that can be obtained from http://abc.org and should be unzipped to the repository main directory.

The core and the toy datasets can be found in the archives ``100k.zip`` and ``toy_set.zip``, available from http://abc.org and http://abc.org, respectively. These should be unzipped to the repository main directory.

## Repository contents description

Core modules:

* ``graph_neural_networks.py`` contains basic methods for constructing graph neural networks. This is the core of the whole repository
* ``graph_utils.py`` contains several utility methods for working with graphs
* ``data_preparation_utils.py`` contains simple methods for preparing data for training (including scaler initialization)
* ``training_and_evaluation.py`` contains methods for training and evaluation of models, reused many times in our experiments
* ``iterative_updaters.py`` contains a few basic iterative updaters for use in gradient descent experiments
* ``toy_examples.py`` currently contains core code for a single toy problem, related to neighbourhood entropy (similar to (2) in t-SNE paper (van der Maaten, L. and Hinton, G. (2008)))  

Experiment scripts:

* ``100k_fit_experiment.py`` contains code for running main experiment with multiple graph nn architectures of type 1 (Table 1 in the paper). Fits models.
* ``100k_eval_experiment.py`` contains code for running main experiment with multiple graph nn architectures of type 1 (Table 1 in the paper). Evaluates models produced by ``100k_fit_experiment.py``.
* ``100k_fit_experiment_kipf.py`` contains code for running main experiment with multiple graph nn architectures of type 2 (Table 2 in paper). Fits models.
* ``100k_eval_experiment_kipf.py`` contains code for running main experiment with multiple graph nn architectures of type 2 (Table 1 in paper). Evaluates models produced by ``100k_fit_experiment_kipf.py``.
* ``100k_fit_experiment_feedforward.py`` contains code for running reference experiment with a few feedforward architectures (Table 3 in paper). Fits models.
* ``100k_eval_experiment_feedforward.py`` contains code for running reference experiment with a few feedforward architectures (Table 3 in paper). Evaluates models produced by ``100k_fit_experiment_feedforward.py``.
* ``random_topologies_experiment.py`` contains code for running random graph experiments for graph nns of type 1 (method 1, Figure 1(a) in the paper).
* ``permuted_topologies_experiment.py`` contains code for running random (permuted) graph experiments for graph nns of type 1 (method 2, Figure 1(b) in the paper).
* ``random_topologies_experiment_kipf.py`` is a analogue of ``random_topologies_experiment.py`` for graph nns of type 2 (check Supplementary materials for results)
* ``permuted_topologies_experiment_kipf.py`` is a analogue of ``permuted_topologies_experiment.py`` for graph nns of type 2 (check Supplementary materials for results)
* ``toy_random_topologies_experiment.py`` is an analogue of ``random_topologies_experiment.py`` for our toy problem (check Supplementary materials for results)
* ``toy_permuted_topologies_experiment.py`` is an analogue of ``permuted_topologies_experiment.py`` for our toy problem (check Supplementary materials for results)

Notebooks:

* ``graph_nn_type_1_tests.ipynb`` is a notebook documenting a number of experiments related to type 1 graph nns, including preliminary results, random graph experiment summary, and some nn visualization (not included in the paper) 
* ``graph_nn_type_2_tests.ipynb`` is an analogue of ``graph_nn_type_1_tests.ipynb`` for type 2 networks.
* ``feedforward_nn_tests.ipynb`` contains first test results for feedforward neural networks, in preparation to produce Table 3 in the paper.
* ``fit_and_evaluate_experiments_summary_type_1.ipynb`` was used for producing Table 1 in the paper, as well as two tables included in Supplementary materials.
* ``fit_and_evaluate_experiments_summary_type_2.ipynb`` is a full analogue of ``fit_and_evaluate_experiments_summary_type_1.ipynb`` for graph nns of type 2. Table 2 was obtained using this code, as well two further tables included in Supplementary materials.
* ``fit_and_evaluate_experiments_summary_feedforward.ipynb`` is a full analogue of ``fit_and_evaluate_experiments_summary_type_1.ipynb`` for fully connected feedforward networks. Table 3 was obtained using this code, as well two further tables included in Supplementary materials.

Result files:

* ``100k_fit_and_evaluate_experiments/fit_eval_results.csv`` contains basic results of the main experiment for type 1 graph nns. These results are now part of Table 1 in the paper.
*  ``100k_fit_and_evaluate_experiments_kipf/fit_eval_results_kipf.csv`` is an analogue of ``100k_fit_and_evaluate_experiments/fit_eval_results.csv`` for type 2 networks. These results are now part of Table 2 in the paper.
* ``100k_feedforward_fit_and_evaluate_experiments/fit_eval_results.csv`` is an analogue of  ``100k_fit_and_evaluate_experiments/fit_eval_results.csv`` for fully connected feedforward nns. These results are now part of Table 3 in the paper. 
* ``random_topologies_3_2.csv`` contains results of random graph experiments for type 1 graph nns (method 1, Figure 1(a) in the paper)
* ``permuted_topologies_0.csv`` contains results of random graph experiments for type 1 graph nns (method 2, Figure 1(b) in the paper)
* ``random_topologies_3_2_kipf.csv`` is an analogue of ``random_topologies_3_2.csv`` for type 2 networks. Used for producing a plot in the Supplementary materials for the paper.
* ``permuted_topologies_0_kipf.csv`` is an analogue of ``permuted_topologies_0.csv`` for type 2 networks. Used for producing a plot in the Supplementary materials for the paper.
* ``toy_random_topologies_0.csv`` contains results of random graph experiments for type 1 graph nns based on our toy problem (graph randomization method 1). Used for producing a plot in the Supplementary materials for the paper.
* ``toy_permuted_topologies_0.csv`` contains results of random graph experiments for type 1 graph nns based on our toy problem (graph randomization method 2). Used for producing a plot in the Supplementary materials for the paper.

Additional:

* ``macierz_sasiedztwa.txt`` contains our traffic optimization problem adjacency matrix. It is a _directed_, nonsymmetric matrix, but we symmetrize it in all our experiments by extracting _undirected_ edges. "Macierz sąsiedztwa" just means "adjacency matrix" in Polish.
* PNG files contain the plots included in the paper and its Supplementary materials.

**NOTE**: Type 2 neural networks were first dubbed "Kipf's networks" by us due to a formal analogy to the networks discussed in (Kipf, T. N. and Welling, M. (2017)). The name stuck, which is visible in several files in the repository.

### Use example

Here is an exemplary (minimalistic) code for constructing a graph neural network of type 1:

```py
import numpy as np
import tensorflow as tf

import graph_utils as graph_utils
import graph_neural_networks as graph_nn

adj_matrix = np.genfromtxt("macierz_sasiedztwa.txt")

nn_input = tf.placeholder(dtype=tf.float32, shape=[None, 21])
nn_output = graph_nn.transfer_matrix_neural_net(nn_input, 3, 4, tf.nn.tanh, adj_matrix, verbose=True)
```

### References

* van der Maaten, L. and Hinton, G. (2008) Visualizing Data using t-SNE, Journal of Machine Learning Research, vol. 9, pp. 2579--2605
* Kipf, T. N. and Welling, M. (2017) Semi-Supervised Classification with Graph Convolutional Networks, 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings