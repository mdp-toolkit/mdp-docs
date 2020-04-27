"""
Nodes for a multilayer perceptron with batch backpropagation learning.

Formulas:
=========
The vector and matrix indices are dropped.

l = 1,..,n layer index

x_0 : input
x_l : output of layer l
t : target output, ideally x_n = t

W_l : weight matrix
s : activation function
s(x) = 1 / (1 + e^{-c*x})   with derivative d/dx s(x) = s(x)*(1 - s(x))

x_l = s(W_l x_{l-1})

D_l : derivative of activation function for last input 
D_l = d/dx s(x_{l-1}) = x_l (1 - x_l)

errors and weight updates:
--------------------------
e_{n+1} = x_l - t
e_l = W_l delta_l

delta_l = D_l e_{l+1}

delta W = - gamma * delta_l x_{l-1}

For batch learning we simply sum over all delta W. 
"""

import numpy as np
import bimdp

# TODO: provide activation function and bias as argument?

class MPerceptronBiNode(bimdp.BiNode):
    """Node for a multilayer perceptron to be trained with backpropagation.
    
    It is used to built up an artificial neural network.
    """
    
    def __init__(self, input_dim=None, output_dim=1, dtype=None,
                 node_id=None):
        """Initialize the node.
        
        output_dim -- Can be > 1 to learn multiple outputs in a single node.
        """
        super(MPerceptronBiNode, self).__init__(input_dim=input_dim,
                                                output_dim=output_dim,
                                                dtype = dtype,
                                                node_id=node_id)
        self._last_x = None
        self._last_y = None
        self._weights = None
        
    def is_trainable(self):
        return False
    
    def is_invertible(self):
        return True
    
    def _is_bi_learning(self):
        return True

    def _execute(self, x):
        """Return the perceptron output.
        
        The output is also stored internally for the backrpopagation learning.
        """
        self._last_x = x
        if self._weights is None:
            self._weights = np.random.random((self.output_dim, self.input_dim))
        self._last_y = 1.0 / (1 + np.exp(np.dot(x, self._weights.T)))
        return self._last_y
    
    # use inverse method for backprop so that switchboard can be used
    def _inverse(self, error, gamma=0.1):
        """Perform batch backprop learning and return the error values."""
        delta = self._last_y * (1 - self._last_y) * error
        new_error = np.dot(delta, self._weights)
        self._weights -= gamma * np.sum(delta[:,:,np.newaxis] *
                                        self._last_x[:,np.newaxis,:],
                                        axis=0)
        return new_error
    
    def _bi_reset(self):
        self._last_x = None
        self._last_y = None
        

class BackpropBiNode(bimdp.BiNode):
    """Node for batch backpropagation learning in a multilayer perceptron.
    
    This node can be used on top of a network of MPerceptronBiNode nodes.
    """
    
    def __init__(self, bottom_node, node_id, input_dim=None, dtype=None):
        """
        bottom_node -- Node id for for node at which the backpropagation
            terminates.
        """
        super(BackpropBiNode, self).__init__(input_dim=input_dim,
                                             output_dim=input_dim,
                                             dtype = dtype,
                                             node_id=node_id)
        self._bottom_node = bottom_node
        self._last_x = None
        
    def is_trainable(self):
        return False
    
    def is_invertible(self):
        return True
    
    def _bi_reset(self):
        self._last_x = None
        
    def _execute(self, x, reference_output=None, gamma=0.1):
        """Start the backpropagation.
        
        x -- output of top layer
        reference_output -- target output for top layer
        gamma -- learning rate
        
        If no reference_output is available this is only the identity.
        """
        if reference_output is not None:
            # unwrap the reference data
            t = reference_output[0]
            self._last_x = x
            error = (x - t)
            msg = {"method": "inverse",
                   "%s->target" % self._bottom_node: self._node_id,
                   "%s->method" % self._node_id: "terminate_backprop",
                   "gamma": gamma}
            return error, msg, -1
        else:
            return x
            
    def _terminate_backprop(self, x, msg):
        """Terminate the backpropagation phase and return the output."""
        del msg["method"]
        return self._last_x, msg
        
