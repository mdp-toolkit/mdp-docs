"""
Same example as in mnist_fda, but this time using BiMDP.
"""

import time
import mdp
import bimdp
import mnistdigits

# TODO: use special task class to expand data remotely

chunk_size = 7000  # for each digit there are about 5000 training samples
verbose = True

flow = bimdp.parallel.ParallelBiFlow([
            mdp.nodes.PCANode(output_dim=50),
            mdp.nodes.PolynomialExpansionNode(degree=2),
            bimdp.nodes.FDABiNode(output_dim=(mnistdigits.N_IDS-1)),
            bimdp.nodes.GaussianBiClassifier()
        ], verbose=verbose)

## training and execution
train_data, train_ids = mnistdigits.get_data("train",
                                             max_chunk_size=chunk_size)
train_msgs = [{"labels": id} for id in train_ids]
test_data, test_ids = mnistdigits.get_data("test", max_chunk_size=chunk_size)
start_time = time.time()
#with mdp.parallel.Scheduler(verbose=verbose) as scheduler:
#with mdp.parallel.ThreadScheduler(n_threads=4, verbose=verbose) as scheduler:
with mdp.parallel.ProcessScheduler(n_processes=4, verbose=verbose) as scheduler:
    flow.train([train_data] * len(flow),
               msg_iterables=[train_msgs] * len(flow),
               scheduler=scheduler)
    y, result_msg = flow.execute(test_data,
                                 [{"return_labels": True}] * len(test_data),
                                 scheduler=scheduler)
total_time = time.time() - start_time
print "time: %.3f secs" % total_time

## analyse the results
result_labels = result_msg["labels"]
n_samples = 0 
n_hits = 0
for i, id_num in enumerate(test_ids):
    chunk_size = len(test_data[i])
    chunk_labels = result_labels[n_samples:(n_samples+chunk_size)]
    n_hits += chunk_labels.count(id_num)
    n_samples += chunk_size
print "performance: %.1f%%" % (100. * n_hits / n_samples)

#mdp.hinet.show_flow(flow)
