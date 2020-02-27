.. _using_mdp_is_as_easy:

Using MDP is as easy as::

    >>> import mdp

    >>> # perform PCA on some data x
    >>> y = mdp.pca(x) # doctest: +SKIP

    >>> # perform ICA on some data x using single precision
    >>> y = mdp.fastica(x, dtype='float32') # doctest: +SKIP 
