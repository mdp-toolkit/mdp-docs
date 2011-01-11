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

- Shogun (http://www.shogun-toolbox.org/):
  The Shogun machine learning toolbox provides a large set of
  different support vector machine implementations and classifiers.
  Each of them can be combined with another large set of kernels.

  The MDP wrapper tries to help with some common default arguments 
  for the kernels and classifiers and will provide some reasonable
  default values when chosing a classifier or a kernel.

  However, in order to avoid problems, users are encouraged to 
  keep an eye on the original C++ API and provide as many parameters
  as specified.

- libsvm (http://www.csie.ntu.edu.tw/~cjlin/libsvm/):
  libsvm is a library for support vector machines. Even though there
  is also a libsvm wrapper in the Shogun toolbox, the libsvm interface
  is a bit simpler to use and the wrapper may also be used to make
  probability estimates.

  Note that starting with MDP 3.0 we only support the Python API
  for the recent libsvm versions 2.91 and 3.0.

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
