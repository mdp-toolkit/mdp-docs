
import numpy as np
import scipy.io
import time

import mdp
import bimdp

# global variables
verbose = True
n_ids = 10
mat_data = scipy.io.loadmat("mnist_all.mat")


#class MNISTIterable(object):
#    
#    def __init__(mat_data, prefix, n_label_chunks):
#        self.prefix = prefix
#        self.n_label_chunks = n_label_chunks
#        
#    def __iter__(self):
#        keys = [self.prefix % i for i in range(n_ids)]
#        for key in keys:
#            chunk_size = len(mat_data[key]) // n_label_chunks
#            for i in range(self.n_label_chunks):
#                yield mat_data[key][i*length:ilength]
#            
            

biflow = bimdp.parallel.ParallelBiFlow([
            mdp.nodes.PCANode(output_dim=15),
            mdp.nodes.PolynomialExpansionNode(degree=2),
            bimdp.nodes.FDABiNode(),
            bimdp.nodes.GaussianBiClassifier()
#            bimdp.nodes.NearestMeanBiClassifier()
         ], verbose=verbose)

train_data = [mat_data["train%d" % i].astype("float32")
              for i in range(n_ids)]
train_msgs = [{"labels": i} for i in range(n_ids)]
test_data = [mat_data["test%d" % i].astype("float32")[:2]
             for i in range(n_ids)]

start_time = time.time()
#with mdp.parallel.ProcessScheduler(verbose=True) as scheduler:
with mdp.parallel.Scheduler(verbose=verbose) as scheduler:
    biflow.train([train_data, None, train_data, train_data],
                 msg_iterables=[None, None, train_msgs, train_msgs],
                 scheduler=scheduler)
    y, result_msg = biflow.execute(test_data,
                                   [{"return_labels": True}] * len(test_data),
                                   scheduler=scheduler)
total_time = time.time() - start_time

result_labels = result_msg["labels"]
n_total = 0 
n_total_hits = 0
for i in range(n_ids):
    n_samples_i = len(test_data[i])
    labels_i = result_labels[n_total:(n_total+n_samples_i)]
    n_total_hits += labels_i.count(i)
    n_total += n_samples_i

if verbose:
    print "==============================================="
print "time: %.3f secs" % total_time
print "performance: %.1f%%" % (100. * n_total_hits / n_total)
