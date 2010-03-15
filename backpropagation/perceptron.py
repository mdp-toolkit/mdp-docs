"""
This file contains everything needed for the binet version of the DBN,
based on Pietros non-binet DBN node implementation.
"""

import numpy as np
import bimdp


class MPerceptronBiNode(bimdp.BiNode):
    """Node for a multilayer perceptron to be trained with backpropagation.
    
    It is used to built up an artificial neural network.
    """
    
    # TODO: optionally add activation function argument and bias?
    def __init__(self, input_dim=None, output_dim=1, dtype=None,
                 node_id=None):
        """Initialize the node.
        
        output_dim -- Can be > 1 to learn multiple outputs in a single node.
        """
        super(MPerceptronBiNode, self).__init__(input_dim=input_dim,
                                                output_dim=output_dim,
                                                dtype = dtype,
                                                node_id=node_id)
        self._backprop_phase = True
        self._last_output = None
        self.weights = None
        
    def is_trainable(self):
        return False
    
    def is_invertible(self):
        return True
    
    def _is_bi_learning(self):
        return self._backprop_phase

    def _execute(self, x):
        if self.weights is None:
            self.weights = np.random.random((self.output_dim, self.input_dim))
        o = 1.0 / (1 + np.exp(np.dot(x, self.weights.T)))
        self._last_output = o
        return o
    
    # use inverse method for backprop so that switchboard can be used
    def _inverse(self, delta, gamma=0.1):
        """
        delta -- backprop error from above layer
        gamma -- 
        """
        if self._backprop_phase and gamma:
            # batch backprop learning
            o = self._last_output
            new_delta = o * (1 - o) * np.dot(delta, self.weights)
            self.weights -= gamma * np.sum(o[:,:,np.newaxis] *
                                           delta[:,np.newaxis,:],
                                           axis=0)
            return new_delta
        else:
            raise bimdp.BiNodeException("Node not in backprop phase.")
    
    def _stop_message(self, stop_backprop=False):
        if stop_backprop:
            self._backprop_phase = False
        
    def bi_reset(self):
        self._last_output = None
        

class BackpropBiNode(bimdp.BiNode):
    """Node for batch backpropagation learning in a multilayer perceptron.
    
    This node can be used on top of a network of MPerceptronBiNode nodes.
    """
    
    # TODO: add _execute flag to shutdown _backprop_phase for all nodes?
    
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
        self._backprop_corout = None
        self._last_output = None
        
    def is_trainable(self):
        return False
    
    def is_invertible(self):
        return True
    
    def bi_reset(self):
        self._last_output = None
        
    def _execute(self, o, reference_output=None, gamma=0.1):
        """Start the backpropagation.
        
        gamma -- learning rate
        """
        if reference_output is not None:
            # unwrap the reference data
            t = reference_output[0]
            self._last_output = o
            # compute backprop error
            delta = o * (1 - o) * (o - t)
            msg = {"method": "inverse",
                   "%s=>target" % self._bottom_node: self._node_id,
                   "%s=>method" % self._node_id: "terminate_backprop",
                   "gamma": gamma}
            return delta, msg, -1
        else:
            return o
            
    def _terminate_backprop(self, x, msg):
        del msg["method"]
        return self._last_output, msg
        
