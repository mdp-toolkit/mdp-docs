"""
Simple demo of the binet DBN version, presenting a training inspection.
"""

# TODO: implement a non-clone layer and uncomment the layers here
# TODO: fix the missing node_id argument in BiFlowNode

# TODO: find a better way to wrap the reference data,
#    change switchboard to only divide if the dimensions exactly fit?
#    or add ignore flag or list with keys to switchboard?

import numpy as np
import bimdp
from perceptron import PerceptronBiNode, BackpropBiNode 


## create simple perceptron
switchboard = bimdp.hinet.BiSwitchboard(input_dim=15, connections=range(15),
                                        node_id="switchboard_1")
#layer = bimdp.hinet.BiLayer([PerceptronBiNode(input_dim=5, output_dim=3)
#                             for _ in range(3)],
#                             node_id="layer_1")
layer = PerceptronBiNode(input_dim=15, output_dim=9, node_id="layer_1")
layer_flownode1 = bimdp.hinet.BiFlowNode(switchboard + layer)
switchboard = bimdp.hinet.BiSwitchboard(input_dim=9, connections=range(9),
                                        node_id="switchboard_2")
#layer = bimdp.hinet.BiLayer([PerceptronBiNode(input_dim=9, output_dim=3)],
#                            node_id="layer_2")
layer = PerceptronBiNode(input_dim=9, output_dim=3, node_id="layer_2")
layer_flownode2 = bimdp.hinet.BiFlowNode(switchboard + layer)
backprop_node = BackpropBiNode(bottom_node="layer_1", node_id="backprop_node")
perceptron = layer_flownode1 + layer_flownode2 + backprop_node

## train
data = np.random.random((100, 15))
# encapsulate reference in dict to not confuse the switchboard
reference = (np.random.random((100, 3)),)
msg = {"reference_output": reference, "gamma": 0.2}
bimdp.show_execution(perceptron, data, msg, debug=True)

# test
data = np.random.random((50, 15))
# encapsulate reference in dict to not confuse the switchboard
bimdp.show_execution(perceptron, data, debug=True)

print "done."