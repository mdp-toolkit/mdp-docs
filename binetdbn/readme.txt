Deep Belief Network (DBN) based on BiMDP
========================================

written by Pietro Berkes and Niko Wilbert

This is only a proof-of-concept, so don't expect the DBN to actually work!
Its current pupose is to experiment with different binet features.

Run the demo_dbn.py file to view an HTML inspection of the DBN training.
This should help with understanding how the implementation works.

Other files:
dbn_nodes.py -- This is the original non-bimdp DBN node from Pietro, which is
    the basis for the binet DBN implementation.
dbn_binodes.py -- The bimdp version of a DBN.
dbn_binodes_coroutine.py -- Almost the same as dbn_binodes.py, but using a
    coroutine for managing the different phases.
  