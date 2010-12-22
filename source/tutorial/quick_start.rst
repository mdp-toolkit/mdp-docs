Quick Start
===========

Using MDP is as easy as

    >>> import mdp
    >>> # perform pca on some data x
    ...
    >>> y = mdp.pca(x) # doctest: +SKIP
    >>> # perform ica on some data x using single precision
    ...
    >>> y = mdp.fastica(x, dtype='float32') # doctest: +SKIP 

MDP requires the numerical Python extensions `NumPy`_ or `SciPy`_. At
import time MDP will select ``scipy`` if available, otherwise
``numpy`` will be loaded. You can force the use of a numerical
extension by setting the environment variable ``MDPNUMX=numpy`` or
``MDPNUMX=scipy``. 

.. admonition:: An important remark
    
   Input array data is typically assumed to be two-dimensional and
   ordered such that observations of the same variable are stored on
   rows and different variables are stored on columns.
