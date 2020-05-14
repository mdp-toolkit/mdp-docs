"""
Simple of a multilayer perceptron implemented with BiMDP.
"""

import numpy as np
import bimdp
from perceptron import MPerceptronBiNode, BackpropBiNode 

## Create simple multilayer perceptron.
# the switchboards are only dummys to show that it works
switchboard1 = bimdp.hinet.BiSwitchboard(input_dim=18, connections=range(18),
                                         node_id="switchboard_1")
# the use_copies flag here means that we use do weight sharing
layer1 = bimdp.hinet.CloneBiLayer(MPerceptronBiNode(input_dim=3, output_dim=2),
                                  n_nodes=6, use_copies=True, node_id="layer_1")
layer_flownode1 = bimdp.hinet.BiFlowNode(switchboard1 + layer1)
switchboard2 = bimdp.hinet.BiSwitchboard(input_dim=12, connections=range(12),
                                         node_id="switchboard_2")
layer2 = bimdp.hinet.CloneBiLayer(MPerceptronBiNode(input_dim=6),
                                  n_nodes=2, use_copies=True, node_id="layer_2")
layer_flownode2 = bimdp.hinet.BiFlowNode(switchboard2 + layer2)
# Note: bottom_node can't be set to container nodes, since the target value
#    gets overridden by the internal node's -1 default inverse target value.
backprop_node = BackpropBiNode(bottom_node="switchboard_1",
                               node_id="backprop_node")
perceptron = layer_flownode1 + layer_flownode2 + backprop_node

## Train with batch backpropagation.
n_patterns = 20
n_training_iterations = 5
data = np.random.random((n_patterns, 18))
# encapsulate reference data in dict to not confuse the switchboard
reference = (np.random.random((n_patterns, 2)),)
msg = {"reference_output": reference, "gamma": 0.2}
# show only the first training iteration
bimdp.show_execution(perceptron, data, msg, debug=True)
# remaining training iterations
perceptron.execute([data for _ in range(n_training_iterations-1)],
                   [msg for _ in range(n_training_iterations-1)])

## Test without backpropagation.
#_, result = bimdp.show_execution(perceptron, data, debug=True)
result = perceptron.execute(data)

print "done."