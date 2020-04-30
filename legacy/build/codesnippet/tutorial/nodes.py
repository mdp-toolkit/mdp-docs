# -*- coding: utf-8 -*-
# Generated by codesnippet sphinx extension on 2020-04-30

import mdp
import numpy as np
np.random.seed(0)
pcanode1 = mdp.nodes.PCANode()
pcanode1
# Expected:
## PCANode(input_dim=None, output_dim=None, dtype=None)

pcanode2 = mdp.nodes.PCANode(output_dim=10)
pcanode2
# Expected:
## PCANode(input_dim=None, output_dim=10, dtype=None)

pcanode3 = mdp.nodes.PCANode(output_dim=0.8)
pcanode3.desired_variance
# Expected:
## 0.8

pcanode4 = mdp.nodes.PCANode(dtype='float32')
pcanode4
# Expected:
## PCANode(input_dim=None, output_dim=None, dtype='float32')

pcanode4.supported_dtypes
# Expected:
## [dtype('float32'), dtype('float64')...]

expnode = mdp.nodes.PolynomialExpansionNode(3)

x = np.random.random((100, 25))  # 25 variables, 100 observations

pcanode1.train(x)

pcanode1
# Expected:
## PCANode(input_dim=25, output_dim=None, dtype='float64')

for i in range(100):
    x = np.random.random((100, 25))
    pcanode1.train(x)

expnode.is_trainable()
# Expected:
## False

pcanode1.stop_training()

pcanode3.train(x)
pcanode3.stop_training()
pcanode3.output_dim
# Expected:
## 16
pcanode3.explained_variance
# Expected:
## 0.85261144755506446

avg = pcanode1.avg            # mean of the input data
v = pcanode1.get_projmatrix() # projection matrix

fdanode = mdp.nodes.FDANode()
for label in ['a', 'b', 'c']:
    x = np.random.random((100, 25))
    fdanode.train(x, label)

fdanode.stop_training()
for label in ['a', 'b', 'c']:
    x = np.random.random((100, 25))
    fdanode.train(x, label)

x = np.random.random((100, 25))
y_pca = pcanode1.execute(x)

y_pca = pcanode1(x)

x = np.random.random((100, 5))
y_exp = expnode(x)

x = np.random.random((100, 25))
y_fda = fdanode(x)

pcanode1.is_invertible()
# Expected:
## True
x = pcanode1.inverse(y_pca)

expnode.is_invertible()
# Expected:
## False

class TimesTwoNode(mdp.Node):
     def is_trainable(self):
         return False
     def _execute(self, x):
         return 2*x
     def _inverse(self, y):
         return y/2
node = TimesTwoNode(dtype = 'float32')
x = mdp.numx.array([[1.0, 2.0, 3.0]])
y = node(x)
print x, '* 2 =  ', y
# Expected:
## [[ 1.  2.  3.]] * 2 =   [[ 2.  4.  6.]]
print y, '/ 2 =', node.inverse(y)
# Expected:
## [[ 2.  4.  6.]] / 2 = [[ 1.  2.  3.]]

class PowerNode(mdp.Node):
    def __init__(self, power, input_dim=None, dtype=None):
        super(PowerNode, self).__init__(input_dim=input_dim, dtype=dtype)
        self.power = power
    def is_trainable(self):
        return False
    def is_invertible(self):
        return False
    def _get_supported_dtypes(self):
        return ['float32', 'float64']
    def _execute(self, x):
        return self._refcast(x**self.power)
node = PowerNode(3)
x = mdp.numx.array([[1.0, 2.0, 3.0]])
y = node(x)
print x, '**', node.power, '=', node(x)
# Expected:
## [[ 1.  2.  3.]] ** 3 = [[  1.   8.  27.]]

class MeanFreeNode(mdp.Node):
    def __init__(self, input_dim=None, dtype=None):
        super(MeanFreeNode, self).__init__(input_dim=input_dim,
                                           dtype=dtype)
        self.avg = None
        self.tlen = 0
    def _train(self, x):
        # Initialize the mean vector with the right
        # size and dtype if necessary:
        if self.avg is None:
            self.avg = mdp.numx.zeros(self.input_dim,
                                      dtype=self.dtype)
        self.avg += mdp.numx.sum(x, axis=0)
        self.tlen += x.shape[0]
    def _stop_training(self):
        self.avg /= self.tlen
        if self.output_dim is None:
            self.output_dim = self.input_dim
    def _execute(self, x):
        return x - self.avg
    def _inverse(self, y):
        return y + self.avg
node = MeanFreeNode()
x = np.random.random((10,4))
node.train(x)
y = node(x)
print 'Mean of y (should be zero):\n', np.abs(np.around(np.mean(y, 0), 15))
# Expected:
## Mean of y (should be zero):
## [ 0.  0.  0.  0.]

class UnitVarianceNode(mdp.Node):
    def __init__(self, input_dim=None, dtype=None):
        super(UnitVarianceNode, self).__init__(input_dim=input_dim,
                                                dtype=dtype)
        self.avg = None # average
        self.std = None # standard deviation
        self.tlen = 0
    def _get_train_seq(self):
        return [(self._train_mean, self._stop_mean),
                (self._train_std, self._stop_std)]
    def _train_mean(self, x):
        if self.avg is None:
            self.avg = mdp.numx.zeros(self.input_dim,
                                      dtype=self.dtype)
        self.avg += mdp.numx.sum(x, 0)
        self.tlen += x.shape[0]
    def _stop_mean(self):
        self.avg /= self.tlen
    def _train_std(self, x):
        if self.std is None:
            self.tlen = 0
            self.std = mdp.numx.zeros(self.input_dim,
                                      dtype=self.dtype)
        self.std += mdp.numx.sum((x - self.avg)**2., 0)
        self.tlen += x.shape[0]
    def _stop_std(self):
        # compute the standard deviation
        self.std = mdp.numx.sqrt(self.std/(self.tlen-1))
    def _execute(self, x):
        return (x - self.avg)/self.std
    def _inverse(self, y):
        return y*self.std + self.avg
node = UnitVarianceNode()
x = np.random.random((10,4))
# loop over phases
for phase in range(2):
    node.train(x)
    node.stop_training()

# execute
y = node(x)
print 'Standard deviation of y (should be one): ', mdp.numx.std(y, axis=0, ddof=1)
# Expected:
## Standard deviation of y (should be one):  [ 1.  1.  1.  1.]

class TwiceNode(mdp.Node):
    def is_trainable(self): return False
    def is_invertible(self): return False
    def _set_input_dim(self, n):
        self._input_dim = n
        self._output_dim = 2*n
    def _set_output_dim(self, n):
        raise mdp.NodeException, "Output dim can not be set explicitly!"
    def _execute(self, x):
        return mdp.numx.concatenate((x, x), 1)
node = TwiceNode()
x = mdp.numx.zeros((5,2))
x
# Expected:
## array([[ 0.,  0.],
##        [ 0.,  0.],
##        [ 0.,  0.],
##        [ 0.,  0.],
##        [ 0.,  0.]])
node.execute(x)
# Expected:
## array([[ 0.,  0.,  0.,  0.],
##        [ 0.,  0.,  0.,  0.],
##        [ 0.,  0.,  0.,  0.],
##        [ 0.,  0.,  0.,  0.],
##        [ 0.,  0.,  0.,  0.]])
