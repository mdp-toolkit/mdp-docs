.. _wrappers:

================================
Interfacing with other libraries
================================
.. codesnippet::


MDP is, of course, not the only Python library to offer an
implementation of signal processing and machine learning methods.
Several other projects, often specialized in different algorithms, or
based on different approaches, are begin developed in parallel. In
order to avoid an excessive duplication of efforts, the long-term
philosophy of MDP is that of to automatically wrap the algorithms
defined in external libraries, if they are available. In this way, MDP
users have access to a larger number of algorithms; at the same
time, we offer the MDP infrastructure (flows, caching, etc.) to
users of the wrapped libraries.

At present, MDP automatically creates wrapper nodes when the following
libraries are installed:

- Shogun (http://www.shogun-toolbox.org/)

- libsvm (http://www.csie.ntu.edu.tw/~cjlin/libsvm/)

- scikits.learn (http://scikit-learn.sourceforge.net/index.html):
  scikits.learn is a collection of efficient machine learning
  algorithms.  We offer automatic wrappers to all algorithms defined
  by in the library scikits.learn, and there are a lot of them!
  The wrapped algorithms can be recognised as their name end
  with ``ScikitsLearnNode``.
  
  All ``ScikitsLearnNode`` contain an instance of the wrapped
  scikits.learn instance in the attribute ``scikits_alg``, and allow
  setting all the parameters using the original keywords. See
  for example ... .

  As of MDP 3.0, the wrappers must be considered experimental, because
  there are still a few inconsistencies in the scikits.learn interface
  that we need to address.
