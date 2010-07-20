
import numpy as np
import bimdp
from newton import NewtonNode


## create the flow
sender_node = bimdp.nodes.SenderBiNode(node_id="sender", recipient_id="newton")
sfa_node = bimdp.nodes.SFABiNode(input_dim=4*4, output_dim=5)
switchboard = bimdp.hinet.Rectangular2dBiSwitchboard(
                                          x_in_channels=8, 
                                          y_in_channels=8,
                                          x_field_channels=4, 
                                          y_field_channels=4,
                                          x_field_spacing=2, 
                                          y_field_spacing=2)
flownode = bimdp.hinet.BiFlowNode(bimdp.BiFlow([sfa_node]))
sfa_layer = bimdp.hinet.CloneBiLayer(flownode,
                                     switchboard.output_channels)
newton_node = NewtonNode(sender_id="sender", input_dim=sfa_layer.output_dim,
                         node_id="newton")
flow = bimdp.BiFlow([sender_node, switchboard, sfa_layer, newton_node])
train_gen = [np.random.random((10, switchboard.input_dim))
             for _ in range(3)]
flow.train([None, None, train_gen, None])

## now do the gradient optimization
# define two random points at starting points and goal points
x = np.random.random((2, switchboard.input_dim))
goal_y = np.random.random((2, newton_node.input_dim))
                          
msg = {"method": "newton", "n_iterations": 3, "start_x": x}

bimdp.show_execution(flow, x=goal_y, msg=msg, target="newton")


print "done."
