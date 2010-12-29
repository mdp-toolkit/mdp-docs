.. _using_mdp_is_as_easy:

Using MDP is as easy as::

    >>> import mdp
    >>> # perform pca on some data x
    ...
    >>> y = mdp.pca(x) # doctest: +SKIP
    >>> # perform ica on some data x using single precision
    ...
    >>> y = mdp.fastica(x, dtype='float32') # doctest: +SKIP 
