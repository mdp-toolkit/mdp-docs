"""
Example for digit recognition with the MNIST dataset.

This example demonstrates how classifiers can be used with BiMDP and is also
another benchmark for parallelization.

closely based on:
Berkes, P. (2005).
Handwritten digit recognition with Nonlinear Fisher Discriminant Analysis.
Proc. of ICANN Vol. 2, Springer, LNCS 3696, 285-287.

To run this example you need the MNIST dataset in Matlab format from
Sam Roweis, available at (about 13 MB large):
http://www.cs.nyu.edu/~roweis/data/mnist_all.mat
"""

import time
import mdp
import mnistdigits

chunk_size = 7000
verbose = True

flow = mdp.parallel.ParallelFlow([
            mdp.nodes.PCANode(output_dim=40),
            mdp.nodes.PolynomialExpansionNode(degree=2),
            mdp.nodes.FDANode(output_dim=(mnistdigits.N_IDS-1)),
            mdp.nodes.GaussianClassifier(execute_method="label")
        ], verbose=verbose)

## training and execution
train_data, train_ids = mnistdigits.get_data("train",
                                             max_chunk_size=chunk_size)
train_labeled_data = zip(train_data, train_ids)
test_data, test_ids = mnistdigits.get_data("test", max_chunk_size=chunk_size)
start_time = time.time()
#with mdp.parallel.Scheduler(verbose=verbose) as scheduler:
#with mdp.parallel.ThreadScheduler(n_threads=4, verbose=verbose) as scheduler:
with mdp.parallel.ProcessScheduler(n_processes=4, verbose=verbose) as scheduler:
    flow.train([train_data, None, train_labeled_data, train_labeled_data],
               scheduler=scheduler)
    result_labels = flow.execute(test_data, scheduler=scheduler)
total_time = time.time() - start_time
print "time: %.3f secs" % total_time

## analyse the results
n_samples = 0 
n_hits = 0
for i, id_num in enumerate(test_ids):
    chunk_size = len(test_data[i])
    chunk_labels = result_labels[n_samples:(n_samples+chunk_size)]
    n_hits += (chunk_labels == id_num).sum()
    n_samples += chunk_size
print "performance: %.1f%%" % (100. * n_hits / n_samples)

#mdp.hinet.show_flow(flow)
