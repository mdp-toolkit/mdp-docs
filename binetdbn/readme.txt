Deep Belief Network (DBN) based on BiMDP
========================================

written by Pietro Berkes and Niko Wilbert

This is only a proof-of-concept, so don't expect the DBN to actually work!
Its current pupose is to experiment with different binet features.

Run the demo_dbn.py file to view an HTML inspection of the DBN training.
This should help with understanding how the implementation works.

Other files:
dbn_nodes.py -- This is the original non-bimdp DBN node from Pietro, which is
    the basis for the BiMDP DBN implementation.
dbn_binodes.py -- The current BiMDP implementation of the DBN,
	based on the coroutine decorator for easy continuation.
	
old_dbn_binodes_coroutine.py -- Older version based on a coroutine, but without
	using the coderoator. This might still be useful to understand how the
	codecorator actually works.
old_dbn_binodes_statemachine.py -- Older version based on a state machine
	implementation.