
import scipy.io

import mdp
import bimdp

# global variables
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
    mdp.nodes.PCANode(output_dim=2),
    mdp.nodes.PolynomialExpansionNode(degree=3),
    bimdp.nodes.FDABiNode(),
    bimdp.nodes.GaussianBiClassifier()
], verbose=True)

scheduler = mdp.parallel.Scheduler(verbose=True)

data = [mat_data["train%d" % i].astype("float32")[:10] for i in range(n_ids)]
msgs = [{"labels": i} for i in range(n_ids)]
biflow.train([data, None, data, data],
             msg_iterables=[None, None, msgs, msgs],
             scheduler=scheduler)

data = [mat_data["test%d" % i].astype("float32")[:10] for i in range(n_ids)]
y, msg = biflow.execute(data, [{"return_labels": "label"}] * len(data),
                        scheduler=scheduler)



print "done."