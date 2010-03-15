"""
Simple demo of the binet DBN version, presenting a training inspection.
"""

# TODO: implement a non-clone layer and uncomment the layers here
# TODO: find a better way to wrap the reference data,
#    change switchboard to only divide if the dimensions exactly fit?
#    or add ignore flag or list with keys to switchboard?

import numpy as np
import bimdp
from perceptron import MPerceptronBiNode, BackpropBiNode 


## create simple perceptron
switchboard = bimdp.hinet.BiSwitchboard(input_dim=18, connections=range(18),
                                        node_id="switchboard_1")
layer = bimdp.hinet.CloneBiLayer(MPerceptronBiNode(input_dim=3),
                                 n_nodes=6, use_copies=True, node_id="layer_1")
layer_flownode1 = bimdp.hinet.BiFlowNode(switchboard + layer)
switchboard = bimdp.hinet.BiSwitchboard(input_dim=6, connections=range(6),
                                        node_id="switchboard_2")
layer = bimdp.hinet.CloneBiLayer(MPerceptronBiNode(input_dim=3),
                                 n_nodes=2, use_copies=True, node_id="layer_2")
layer_flownode2 = bimdp.hinet.BiFlowNode(switchboard + layer)
# Note: bottom_node can't be set to container nodes, since the target value
#    gets overridden by the internal node's -1 default inverse target value.
backprop_node = BackpropBiNode(bottom_node="switchboard_1",
                               node_id="backprop_node")
perceptron = layer_flownode1 + layer_flownode2 + backprop_node

## train
data = np.random.random((100, 18))
# encapsulate reference data in dict to not confuse the switchboard
reference = (np.random.random((100, 2)),)
msg = {"reference_output": reference, "gamma": 0.2}
bimdp.show_execution(perceptron, data, msg, debug=True)

# test
data = np.random.random((50, 18))
_, result = bimdp.show_execution(perceptron, data, debug=True)

print "done."