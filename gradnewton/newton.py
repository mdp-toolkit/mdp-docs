
import numpy as np
import mdp
import bimdp


class NewtonNode(bimdp.nodes.IdentityBiNode):
    """Node to implement gradient descent with the Newton method."""
    
    def __init__(self, sender_id, **kwargs):
        """Init.
        
        sender_id -- Id of the SenderBiNode that is at the front of the flow,
            or at the point where the x value should be optimized.
        """
        self.sender_id = sender_id
        super(NewtonNode, self).__init__(**kwargs)

    @bimdp.binode_coroutine(["grad", "msg_x", "msg"])
    def _newton(self, y_goal, n_iterations, x_start, msg):
        """Try to reach the given y value with gradient descent.
        
        The Newton method is used to calculate the next point.
        """
        mdp.activate_extension("gradient")
        # get the y value for the output
        msg = {self.node_id + "->method": "newton", "method": "gradient"}
        y, grad, x, _ = yield x_start, msg.copy(), self.sender_id
        for _ in range(n_iterations):
            # use Newton's method to get the new data point
            error = np.sum((y - y_goal) ** 2, axis=1)
            error_grad = np.sum(2 * (y - y_goal)[:,:,np.newaxis] * grad,
                                axis=1)
            err_grad_norm = np.sqrt(np.sum(error_grad**2, axis=1))
            unit_error_grad = error_grad / err_grad_norm[:,np.newaxis]
            # x_{n+1} = x_n - f(x_n) / f'(x_n)
            x = x - (error / err_grad_norm)[:,np.newaxis] * unit_error_grad
            y, grad, x, _ = yield x, msg.copy(), self.sender_id
        raise StopIteration(x, None, "exit")
        mdp.deactivate_extension("gradient")
