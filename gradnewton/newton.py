
import mdp
import bimdp


class NewtonNode(bimdp.nodes.IdentityBiNode):
    
    def __init__(self, sender_id, **kwargs):
        self.sender_id = sender_id
        super(NewtonNode, self).__init__(**kwargs)

    @bimdp.binode_coroutine(["x", "grad", "msg_x"])    
    def _newton(self, goal_y, n_iterations, start_x):
        mdp.activate_extension("gradient")
        # get the y value for the output
        msg = {self.node_id + "->method": "newton", "method": "gradient"}
        y, grad, x = yield start_x, msg, self.sender_id
        
        mdp.deactivate_extension("gradient")
