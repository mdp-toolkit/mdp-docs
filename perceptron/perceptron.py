"""
This file contains everything needed for the binet version of the DBN,
based on Pietros non-binet DBN node implementation.
"""

import numpy as np
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
        return x[:,:self.output_dim]
    
    # use inverse method for backprop so that switchboard can be used
    def _inverse(self, x):
        """
        x -- error during backrpop phase
        """
        if self._backprop_phase:
            # TODO: implement
            return np.random.random((len(x), self.input_dim))
        else:
            raise Exception()
    
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
        
    def _execute(self, x, reference_output=None, gamma=None):
        """Start the backpropagation.
        
        gamma -- Learning rate, also serves as flag for backpropagation.
        """
        if reference_output is not None:
            # unwrap the reference data
            reference_output = reference_output[0]
            self._last_output = x
            # TODO: implement
            error = x - reference_output
        else:
            error = None
        if gamma is not None:
            if error is None:
                raise Exception()
            msg = {"method": "inverse",
                   "%s=>target" % self._bottom_node: self._node_id,
                   "%s=>method" % self._node_id: "terminate_backprop",
                   # TODO: avoid having to wrap the error
                   "final_error": (error,)}
            return error, msg, -1
        else:
            return 
            
    def _terminate_backprop(self, x, msg):
        del msg["method"]
        return self._last_output, msg
        
