"""
A minimal example for the bimdp coroutine decorator.
"""

import numpy as np
import bimdp


class SimpleCoroutineNode(bimdp.nodes.IdentityBiNode):
    
    # the arg ["b"] means that that the signature will be (x, b)
    @bimdp.binode_coroutine(["b"])
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
x, msg = node.execute(x, {"n_iterations": n_iterations})
print msg  # leftover msg
# the following calls go to the yield statement,
# finally the bs are returned
for i in range(n_iterations-1):
    x, msg = node.execute(x, {"b": i})
    print msg
x, msg = node.execute(x, {"b": n_iterations-1})
print msg
