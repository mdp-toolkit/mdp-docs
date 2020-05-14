import mdp
import numpy
import pylab

import sklearn as sl
from sklearn import datasets

digits = datasets.load_digits()
images = digits.images.astype('f')
labels = digits.target
data = digits.images.reshape((images.shape[0],
                              numpy.prod(images.shape[1:])))

# --- prepare train and test data
# number of digits to be used as training data
ntrain = images.shape[0] // 3 * 2
train_data = [data[:ntrain, :]]
train_data_with_labels = [(data[:ntrain, :], labels[:ntrain])]
test_data = data[ntrain:, :]
test_labels = labels[ntrain:]

# --- build MDP pipeline, last node is a scikit's SVC

# the pipeline is as follow:
# - reduce the dimensionality of the data to 25
# - expand the data in the space of polynomials of deg 3 to get nonlinear FDA
# - only keep the 99% of the variance in the very high-dim polynomial space
# - perform Fished Discriminant Analysis
# - SVC on the resulting space

# the data type is set to float32 to spare memory

flow = mdp.Flow([mdp.nodes.PCANode(output_dim=25, dtype='f'),
                 mdp.nodes.PolynomialExpansionNode(3),
                 mdp.nodes.PCANode(output_dim=0.99),
                 mdp.nodes.FDANode(output_dim=9),
                 mdp.nodes.SVCScikitsLearnNode(kernel='rbf')], verbose=True)

# set the execution behavior of the last node to return labels
#flow[-1].execute = flow[-1].label

# --- train MDP pipeline
flow.train([train_data, None, train_data,
            train_data_with_labels, train_data_with_labels])

# one can have a look at the final state of the nodes this way:
print repr(flow)

# --- get test labels and compute percent error
# get test labels
prediction = flow(test_data)
# percent error
error = ((prediction.flatten() != test_labels).astype('f').sum()
         / (images.shape[0] - ntrain) * 100.)
print 'percent error:', error
