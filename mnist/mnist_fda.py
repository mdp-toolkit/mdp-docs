"""
Example for digit recognition with the MNIST dataset.

This example demonstrates how classifiers can be used with BiMDP and is also
another benchmark for parallelization.

closely based on: 
Berkes, P. (2005).
Handwritten digit recognition with Nonlinear Fisher Discriminant Analysis.
Proc. of ICANN Vol. 2, Springer, LNCS 3696, 285-287.

To run this example you need the MNIST dataset in Matlab format from
Sam Roweis, available at:
http://www.cs.nyu.edu/~roweis/data/mnist_all.mat
"""

import numpy as np
import scipy.io
import time

import mdp
import bimdp

# TODO: use special job class to expand data remotely

## global variables / parameters
n_ids = 10
mat_data = scipy.io.loadmat("mnist_all.mat")
chunk_size = 5000  # for each digit there are about 5000 training samples
verbose = False

biflow = bimdp.parallel.ParallelBiFlow([
            mdp.nodes.PCANode(output_dim=35),
            mdp.nodes.PolynomialExpansionNode(degree=2),
            bimdp.nodes.FDABiNode(output_dim=(n_ids-1)),
            bimdp.nodes.GaussianBiClassifier()
#            bimdp.nodes.NearestMeanBiClassifier()
         ], verbose=verbose)

## prepare data
train_data = []
train_msgs = []
for id in range(n_ids):
    id_key = "train%d" % id
    n_chunks = int(np.ceil(len(mat_data[id_key]) / float(chunk_size)))
    train_data += [mat_data[id_key][i_chunk*chunk_size :
                                    (i_chunk+1)*chunk_size].astype("float32")
                   for i_chunk in range(n_chunks)]
    train_msgs += [{"labels": id} for i in range(n_chunks)]
test_data = []
for id in range(n_ids):
    id_key = "test%d" % id
    n_chunks = int(np.ceil(len(mat_data[id_key]) / float(chunk_size)))
    test_data += [mat_data[id_key][i_chunk*chunk_size :
                                    (i_chunk+1)*chunk_size].astype("float32")
                   for i_chunk in range(n_chunks)]

## training and execution
start_time = time.time()
with mdp.parallel.ThreadScheduler(n_threads=4, verbose=verbose) as scheduler:
#with mdp.parallel.Scheduler(verbose=verbose) as scheduler:
    biflow.train([train_data, None, train_data, train_data],
                 msg_iterables=[None, None, train_msgs, train_msgs],
                 scheduler=scheduler)
    y, result_msg = biflow.execute(test_data,
                                   [{"return_labels": True}] * len(test_data),
                                   scheduler=scheduler)
total_time = time.time() - start_time
if verbose:
    print "==============================================="
print "time: %.3f secs" % total_time

## analyse the results
result_labels = result_msg["labels"]
n_total = 0 
n_total_hits = 0
for id in range(n_ids):
    n_samples_i = len(mat_data["test%d" % id])
    labels_i = result_labels[n_total:(n_total+n_samples_i)]
    n_total_hits += labels_i.count(id)
    n_total += n_samples_i
print "performance: %.1f%%" % (100. * n_total_hits / n_total)
