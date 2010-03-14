"""
This file contains everything needed for the binet version of the DBN,
based on Pietros non-binet DBN node implementation.
"""

import numpy as np

import mdp
import bimdp

class PerceptronBiNode(bimdp.BiNode):
    """Adapter to turn the DBNLayerNode into a BiNode."""

    def __init__(self, input_dim=None, output_dim=None, dtype=None,
                 node_id=None):
        super(PerceptronBiNode, self).__init__(input_dim=input_dim,
                                               output_dim=output_dim,
                                               dtype = dtype,
                                               node_id=node_id)
        self._backprop_phase = True
        self._last_output = None
        
    def is_trainable(self):
        return False
    
    def is_invertible(self):
        return True
    
    def _is_bi_learning(self):
        return self._backprop_phase

    def _execute(self, x):
        # TODO: implement
        return x
    
    # use inverse method for backprop so that switchboard can be used
    def _inverse(self, x):
        """
        x -- error during backrpop phase
        """
        if self._backprop_phase:
            # TODO: implement
            return x
        else:
            # could implement some kind of bayesian mechanism
            return x
    
    def _stop_message(self, stop_backprop=False):
        if stop_backprop:
            self._backprop_phase = False
        
    def bi_reset(self):
        self._last_output = None
        
        
class BackpropBiNode(bimdp.BiNode):
    
    # TODO: add _execute flag to shutdown _backprop_phase for all nodes?
    
    def __init__(self, bottom_node, node_id, input_dim=None, dtype=None):
        """
        bottom_node -- Node id for for node at which the backpropagation
            terminates.
        """
        super(PerceptronBiNode, self).__init__(input_dim=input_dim,
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
        
    # TODO: use gamma as flag instead of backprop_train?
    def _execute(self, reference_output=None, gamma=1.0, backprop_train=False):
        """Start the backpropagation.
        
        gamma -- Learning rate.
        """
        if reference_output:
            # TODO: implement
            error = 1.0
        else:
            error = None
        if backprop_train:
            if error is None:
                raise Exception()
            msg = {"method": "inverse",
                   "%s=>target" % self._bottom_node: self._node_id,
                   "%s=>method" % self._node_id: "terminate_backprop",
                   "final_error": error}
            return error, msg, -1
        else:
            return 
            
    def _terminate_backprop(self, msg):
        del msg["method"]
        return self._last_output, msg
        
