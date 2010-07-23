"""
A minimal example for the bimdp coroutine decorator.
"""

import numpy as np
import bimdp


class SimpleCoroutineNode(bimdp.nodes.IdentityBiNode):
    """Just a stupid example."""
    
    # ["b"] means that yield will return a tuple with x (which is always
    # the first value) and "a" and "b" from the msg
    
    @bimdp.binode_coroutine(["b", "c"])
    def _execute(self, x, n_iterations):
        """Gather all the incomming b and return them finally."""
        bs = []
        for _ in range(n_iterations):
            x, b = yield x
            bs.append(b)
        raise StopIteration(x, {"all the b": bs}) 


n_iterations = 3
x = np.random.random((1,1))
node = SimpleCoroutineNode()

# during the first call the decorator creates the actual coroutine
print node.execute(x, {"n_iterations": n_iterations})

# the following calls no go to the yield statement,
# finally the arguments of the StopIteration are returned
for i in range(n_iterations):
    print node.execute(x, {"b": i})

    
