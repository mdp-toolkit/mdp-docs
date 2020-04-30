"""
Same example as in mnist_fda, but this time using BiMDP and hinet.

However, the performance is worse than for the simpler models in mnist_fda or
mnist_bifda, so this is only a proof-of-concept.
"""

import time
import mdp
import bimdp
import mnistdigits

chunk_size = 2000
verbose = True

pca_dim = 35
fda1_dim = 9
layer1_switchboard = mdp.hinet.Rectangular2dSwitchboard(in_channels_xy=28,
                                                        field_channels_xy=14,
                                                        field_spacing_xy=14)
layer1_node = bimdp.hinet.BiFlowNode(bimdp.BiFlow([
                mdp.nodes.PCANode(input_dim=14**2, output_dim=pca_dim),
                mdp.nodes.QuadraticExpansionNode(),
                bimdp.nodes.FDABiNode(output_dim=fda1_dim)
              ]))
biflow = bimdp.parallel.ParallelBiFlow([
            layer1_switchboard,
            bimdp.hinet.CloneBiLayer(layer1_node, n_nodes=4),
#            mdp.nodes.PCANode(output_dim=pca_dim),
            mdp.nodes.QuadraticExpansionNode(),
            bimdp.nodes.FDABiNode(output_dim=(mnistdigits.N_IDS)),
            bimdp.nodes.GaussianBiClassifier()
         ], verbose=verbose)

## training and execution
train_data, train_ids = mnistdigits.get_data("train",
                                             max_chunk_size=chunk_size)
train_msgs = [{"labels": id} for id in train_ids]
test_data, test_ids = mnistdigits.get_data("test", max_chunk_size=chunk_size)
start_time = time.time()
with mdp.parallel.Scheduler(verbose=verbose) as scheduler:
#with mdp.parallel.ThreadScheduler(n_threads=4, verbose=verbose) as scheduler:
#with mdp.parallel.ProcessScheduler(n_processes=4, verbose=verbose) as scheduler:
    biflow.train([train_data] * len(biflow),
                 msg_iterables=[train_msgs] * len(biflow),
                 scheduler=scheduler)
    y, result_msg = biflow.execute(test_data,
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

#mdp.hinet.show_flow(biflow)
