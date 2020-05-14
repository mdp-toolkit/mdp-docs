"""
Simplified version of mnist_fda, which is used in the MDP paper.
"""

import mdp
import mnistdigits  # helper module for digit dataset

# Create the nodes and combine them in flow.
flow = mdp.parallel.ParallelFlow([
            mdp.nodes.PCANode(output_dim=40),
            mdp.nodes.PolynomialExpansionNode(degree=2),
            mdp.nodes.FDANode(output_dim=(mnistdigits.N_IDS-1)),
            mdp.nodes.GaussianClassifier(execute_method="label")
       ])
# Prepare training and test data.
train_data, train_ids = mnistdigits.get_data("train")
train_labeled_data = zip(train_data, train_ids)
train_iterables = [train_data, None,
                   train_labeled_data, train_labeled_data] 
test_data, test_ids = mnistdigits.get_data("test")
# Parallel training and execution.
with mdp.parallel.ProcessScheduler() as scheduler:
    flow.train(train_iterables, scheduler=scheduler)
    result_labels = flow.execute(test_data, scheduler=scheduler)
# Analysis of the results.
n_samples = 0 
n_hits = 0
for i, id_num in enumerate(test_ids):
    chunk_size = len(test_data[i])
    chunk_labels = result_labels[n_samples:(n_samples+chunk_size)]
    n_hits += (chunk_labels == id_num).sum()
    n_samples += chunk_size
print "performance: %.1f%%" % (100. * n_hits / n_samples)
