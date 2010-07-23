*******************
Nodes and Utilities
*******************

.. _node-list:

Node List
=========
Here is the complete list of implemented nodes.
Refer to the
`API <http://mdp-toolkit.sourceforge.net/docs/api/index.html>`_
for the full documentation and interface description.

:api:`mdp.nodes.AdaptiveCutoffNode`

    Works like the `HistogramNode`. The cutoff bounds are then chosen such     that a given fraction of the training data would have been clipped.

:api:`mdp.nodes.CuBICANode`
 
    Perform Independent Component Analysis using the CuBICA algorithm.

    Reference: Blaschke, T. and Wiskott, L. (2003).
    *CuBICA: Independent Component Analysis by Simultaneous Third- and
    Fourth-Order Cumulant Diagonalization*.
    IEEE Transactions on Signal Processing, 52(5), pp. 1250-1256.
    More information about ICA can be found among others in
    Hyvarinen A., Karhunen J., Oja E. (2001). *Independent Component Analysis*, Wiley.

:api:`mdp.nodes.CutoffNode`

    Clip the data at the specified upper and lower bounds.

:api:`mdp.nodes.DiscreteHopfieldClassifier`

    Learns discrete patterns and can retrieve them again even when they are slightly distorted.

:api:`mdp.nodes.EtaComputerNode`

    Compute the eta values of the normalized training data.
    The delta value of a signal is a measure of its temporal
    variation, and is defined as the mean of the derivative squared,
    i.e. ``delta(x) = mean(dx/dt(t)^2)``. ``delta(x)`` is zero if
    'x' is a constant signal, and increases if the temporal variation
    of the signal is bigger.
    The eta value is a more intuitive measure of temporal variation,
    defined as ``eta(x) = T/(2*pi) * sqrt(delta(x))``.
    If 'x' is a signal of length 'T' which consists of a sine function
    that accomplishes exactly 'N' oscillations, then ``eta(x) = N``.
   
    Reference: Wiskott, L. and Sejnowski, T.J. (2002).
    *Slow Feature Analysis:
    Unsupervised Learning of Invariances*, Neural Computation,
    14(4):715-770.

:api:`mdp.nodes.FANode`

    Perform Factor Analysis. The current implementation should be most
    efficient for long data sets: the sufficient statistics are
    collected in the training phase, and all EM-cycles are performed at
    its end. More information about Factor Analysis can be found in
    `Max Welling's classnotes<http://www.ics.uci.edu/~welling/classnotes/classnotes.html>`_ in the chapter "Linear Models".

:api:`mdp.nodes.FastICANode`

    Perform Independent Component Analysis using the FastICA algorithm.
   
    Reference: Aapo Hyvarinen (1999).
    *Fast and Robust Fixed-Point Algorithms for Independent Component Analysis*,
    IEEE Transactions on Neural Networks, 10(3):626-634.
    More information about ICA can be found among others in
    Hyvarinen A., Karhunen J., Oja E. (2001). *Independent Component Analysis*,
    Wiley.

:api:`mdp.nodes.FDANode`

    Perform a (generalized) Fisher Discriminant Analysis of its
    input. It is a supervised node that implements FDA using a
    generalized eigenvalue approach.
   
    More information on Fisher Discriminant Analysis can be found for
    example in C. Bishop, *Neural Networks for Pattern Recognition*,
    Oxford Press, pp. 105-112.

:api:`mdp.nodes.GaussianClassifierNode`

    Perform a supervised Gaussian classification.  Given a set of
    labelled data, the node fits a gaussian distribution to each
    class.

:api:`mdp.nodes.GrowingNeuralGasNode`

    Learn the topological structure of the input data by building a corresponding
    graph approximation. 
   
    More information about the Growing Neural Gas algorithm can be found in B.
    Fritzke, *A Growing Neural Gas Network Learns Topologies*, in G. Tesauro, D. S.
    Touretzky, and T. K. Leen (editors), *Advances in Neural Information
    Processing Systems 7*, pages 625-632. MIT Press, Cambridge MA, 1995.

:api:`mdp.nodes.HistogramNode`

    Store a fraction of the incoming data during training. This data can then
    be used to analyse the histogram of the data.

:api:`mdp.nodes.HitParadeNode`

    Collect the first 'n' local maxima and minima of the training signal
    which are separated by a minimum gap 'd'.

:api:`mdp.nodes.HLLENode`

    Original code contributed by Jake VanderPlas.

    Perform a Hessian Locally Linear Embedding analysis on the data.
                              
    Implementation based on algorithm outlined in
    David L. Donoho and Carrie Grimes, 
    *Hessian Eigenmaps: new locally linear embedding techniques
    for high-dimensional data*, Proceedings of the National Academy of Sciences
    100(10):5591-5596 (2003).

:api:`mdp.nodes.ISFANode`

    Perform Independent Slow Feature Analysis on the input data.
   
    More information about ISFA can be found in:
    Blaschke, T. , Zito, T., and Wiskott, L.
    *Independent Slow Feature Analysis and Nonlinear Blind Source Separation.*
    Neural Computation 19(4):994-1021 (2007).

:api:`mdp.nodes.JADENode`

    Original code contributed by Gabriel Beckers.

    Perform Independent Component Analysis using the JADE algorithm.

    References:
    Cardoso, J.-F, and Souloumiac, A.
    *Blind beamforming for non Gaussian signals.*
    Radar and Signal Processing, IEE Proceedings F, 140(6): 362-370 (1993), and
    Cardoso, J.-F.
    *High-order contrasts for independent component analysis.*
    Neural Computation, 11(1): 157-192 (1999).   
    More information about ICA can be found among others in
    Hyvarinen A., Karhunen J., Oja E. (2001). *Independent Component Analysis*,
    Wiley.

:api:`mdp.nodes.KMeansClassifier`

    Employs K-Means Clustering for a given number of centroids.

:api:`mdp.nodes.LibSVMClassifier`

    The LibSVMClassifier class acts as a wrapper around the LibSVM library for support vector machines, which needs to be installed as a python module. The software can be found `here<http://www.csie.ntu.edu.tw/~cjlin/libsvm/>`_

    **Warning**: Because it is a new 
    addition to MDP, the LibSVMClassifier should be used with caution. Also, the
    interface might have some flaws. Any hints or bug reports are very welcome.

    See also,
    Chih-Chung Chang and Chih-Jen Lin, *LIBSVM : a library for support vector machines* (2001). 

:api:`mdp.nodes.LinearRegressionNode`

    Compute least-square, multivariate linear regression on the input data.

:api:`mdp.nodes.LLENode`

    Original code contributed by Jake VanderPlas.

    Perform a Locally Linear Embedding analysis on the data.
                             
    Based on the algorithm outlined in *An Introduction to Locally
    Linear Embedding* by L. Saul and S. Roweis, using improvements
    suggested in *Locally Linear Embedding for Classification* by
    D. deRidder and R.P.W. Duin.
   
    References: Sam Roweis and Lawrence Saul, *Nonlinear dimensionality reduction by locally linear embedding*, Science 290(5500):2323-2326, 2000.

:api:`mdp.nodes.NIPALSNode`

    Original code contributed by Michael Schmuker, Susanne Lezius, and Farzad Farkhooi.

    Perform Principal Component Analysis using the NIPALS algorithm.
    This algorithm is particularyl useful if you have more variable than
    observations, or in general when the number of variables is huge and
    calculating a full covariance matrix may be unfeasable. It's also more
    efficient of the standard PCANode if you expect the number of significant
    principal components to be a small. In this case setting output_dim to be
    a certain fraction of the total variance, say 90%, may be of some help.

    Reference for NIPALS (Nonlinear Iterative Partial Least Squares):
    Wold, H.
    *Nonlinear estimation by iterative least squares procedures.*
    in David, F. (Editor), Research Papers in Statistics, Wiley,
    New York, pp 411-444 (1966).
   
    More information about Principal Component Analysis, a.k.a. discrete
    Karhunen-Loeve transform can be found among others in
    I.T. Jolliffe, *Principal Component Analysis*, Springer-Verlag (1986).

:api:`mdp.nodes.NoiseNode`

    Original code contributed by Mathias Franzius.
   
    Inject multiplicative or additive noise into the input data.

:api:`mdp.nodes.PCANode`

    Filter the input data throug the most significatives of its
    principal components.
 
    More information about Principal Component Analysis, a.k.a. discrete
    Karhunen-Loeve transform can be found among others in
    I.T. Jolliffe, *Principal Component Analysis*, Springer-Verlag (1986).

:api:`mdp.nodes.PerceptronClassifier`

    Trains a single binary perceptron with multiple inputs.

:api:`mdp.nodes.PolynomialExpansionNode`

    Perform expansion in a polynomial space.

:api:`mdp.nodes.QuadraticExpansionNode`

    Perform expansion in the space formed by all linear and quadratic
    monomials.

:api:`mdp.nodes.RBMNode`

    Implementation of a Restricted Boltzmann Machine.

    For more information on RBMs, see
    Geoffrey E. Hinton (2007) `Boltzmann machine.
    <http://www.scholarpedia.org/article/Boltzmann_machine>`_
    Scholarpedia, 2(5):1668


:api:`mdp.nodes.RBMWithLabelsNode`

    Implementation of a Restricted Boltzmann Machine with softmax labels.

    For more information on RBMs, see
    Geoffrey E. Hinton (2007) `Boltzmann machine
    <http://www.scholarpedia.org/article/Boltzmann_machine>`_
    Scholarpedia, 2(5):1668

    Hinton, G. E, Osindero, S., and Teh, Y. W. *A fast learning
    algorithm for deep belief nets*, Neural Computation, 18:1527-1554 (2006). 
   
:api:`mdp.nodes.ShogunSVMClassifier`

    The ShogunSVMClassifier class works as a wrapper class for accessing the
    SHOGUN machine learning toolbox. We use the python_modular wrapper to access SHOGUN
    and SHOGUN must not be older than version 0.9. **Warning**: Because it is a new 
    addition to MDP, the ShogunSVMClassifier should be used with caution. Also, the
    interface might have some flaws. Any hints or bug reports are very welcome.

    Most of the kernel machines and linear classifiers of shogun should work with
    this class.

    For exact information about data formats which SHOGUN can accept, see
    http://www.shogun-toolbox.org/

    S. Sonnenburg, G. Raetsch, C. Schaefer and B. Schoelkopf, *Large Scale Multiple Kernel
    Learning*, Journal of Machine Learning Research, 7:1531-1565 (2006).

:api:`mdp.nodes.SFANode`

    Extract the slowly varying components from the input data.
 
    More information about Slow Feature Analysis can be found in
    Wiskott, L. and Sejnowski, T.J., *Slow Feature Analysis: Unsupervised
    Learning of Invariances*, Neural Computation, 14(4):715-770 (2002).

:api:`mdp.nodes.SFA2Node`

    Get an input signal, expand it in the space of
    inhomogeneous polynomials of degree 2 and extract its slowly varying
    components. The ``get_quadratic_form`` method returns the input-output
    function of one of the learned unit as a ``mdp.utils.QuadraticForm`` object.

    More information about Slow Feature Analysis can be found in
    Wiskott, L. and Sejnowski, T.J., *Slow Feature Analysis: Unsupervised
    Learning of Invariances*, Neural Computation, 14(4):715-770 (2002).

:api:`mdp.nodes.SimpleMarkovClassifier`

    Learns the probability with which a label is assigned to a label.

:api:`mdp.nodes.TDSEPNode`

    Perform Independent Component Analysis using the TDSEP algorithm.
    Note that TDSEP, as implemented in this Node, is an online algorithm,
    i.e. it is suited to be trained on huge data sets, provided that the
    training is done sending small chunks of data for each time.

    Reference:
    Ziehe, Andreas and Muller, Klaus-Robert (1998).
    *TDSEP an efficient algorithm for blind separation using time structure.*
    in Niklasson, L, Boden, M, and Ziemke, T (Editors), Proc. 8th Int. Conf. 
    Artificial Neural Networks (ICANN 1998).

:api:`mdp.nodes.TimeFramesNode`

    Copy delayed version of the input signal on the space dimensions.
    ::

       For example, for time_frames=3 and gap=2: 
    
       [ X(1) Y(1)        [ X(1) Y(1) X(3) Y(3) X(5) Y(5)
         X(2) Y(2)          X(2) Y(2) X(4) Y(4) X(6) Y(6)
         X(3) Y(3)   -->    X(3) Y(3) X(5) Y(5) X(7) Y(7)
         X(4) Y(4)          X(4) Y(4) X(6) Y(6) X(8) Y(8)
         X(5) Y(5)          ...  ...  ...  ...  ...  ... ]
         X(6) Y(6)
         X(7) Y(7)
         X(8) Y(8)
         ...  ...  ]

:api:`mdp.nodes.WhiteningNode`

    'Whiten' the input data by filtering it through the most
    significatives of its principal components. All output
    signals have zero mean, unit variance and are decorrelated.

:api:`mdp.nodes.XSFANode`

    Perform Non-linear Blind Source Separation using Slow Feature Analysis.
    This node is designed to iteratively extract statistically
    independent sources from (in principle) arbitrary invertible
    nonlinear mixtures. The method relies on temporal correlations in
    the sources and consists of a combination of nonlinear SFA and a
    projection algorithm. More details can be found in the reference
    given below (once it's published).
   
    More information about XSFA can be found in:
    Sprekeler, H., Zito, T., and Wiskott, L. (2009).
    *An Extension of Slow Feature Analysis for Nonlinear Blind Source Separation.*
    Journal of Machine Learning Research, under revision.
      
.. admonition:: Didn't you find what you were looking for?
   
    If you want to contribute some code or a new
    algorithm, please do not hesitate to submit it!


Additional utilities
====================

MDP offers some additional utilities of general interest
in the ``mdp.utils`` module. Refer to the
`API <http://mdp-toolkit.sourceforge.net/docs/api/index.html>`_
for the full documentation and interface description.

:api:`mdp.utils.CovarianceMatrix`
     This class stores an empirical covariance matrix that can be updated
     incrementally. A call to the ``fix`` method returns the current state
     of the covariance matrix, the average and the number of observations,
     and resets the internal data.

     Note that the internal sum is a standard ``__add__`` operation. We are not
     using any of the fancy sum algorithms to avoid round off errors when
     adding many numbers. If you want to contribute a ``CovarianceMatrix``
     class that uses such algorithms we would be happy to include it in
     MDP.  For a start see the `Python recipe
     <http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/393090>`_
     by Raymond Hettinger. For a
     review about floating point arithmetic and its pitfalls see
     this `interesting article <http://docs.sun.com/source/806-3568/ncg_goldberg.html>`_.

:api:`mdp.utils.DelayCovarianceMatrix`

     This class stores an empirical covariance matrix between the signal and
     time delayed signal that can be updated incrementally.

:api:`mdp.utils.MultipleCovarianceMatrices`

     Container class for multiple covariance matrices to easily
     execute operations on all matrices at the same time.
    
:api:`mdp.utils.dig_node` (node)
    Crawl recursively an MDP ``Node`` looking for arrays.
    Return (dictionary, string), where the dictionary is:
    { attribute_name: (size_in_bytes, array_reference)}
    and string is a nice string representation of it.

:api:`mdp.utils.get_node_size` (node)
    Get 'node' total byte-size using ``cPickle`` with protocol=2.
    (The byte-size is related the memory needed by the node).

:api:`mdp.utils.progressinfo` (sequence, length, style, custom)
    A fully configurable text-mode progress info box tailored to the 
    command-line die-hards.
    To get a progress info box for your loops use it like this::

         >>> for i in progressinfo(sequence):
         ...     do_something(i)

    You can also use it with generators, files or any other iterable object,
    but in this case you have to specify the total length of the sequence::
 
        >>> for line in progressinfo(open_file, nlines):
        ...     do_something(line)
          

    A few examples of the available layouts:: 
 
        [===================================73%==============>...................]

        Progress:  67%[======================================>                   ]

        23% [02:01:28] - [00:12:37]

:api:`mdp.utils.QuadraticForm`

    Define an inhomogeneous quadratic form as ``1/2 x'Hx + f'x + c``.
    This class implements the quadratic form analysis methods
    presented in:
    Berkes, P. and Wiskott, L. On the analysis and interpretation
    of inhomogeneous quadratic forms as receptive fields. *Neural
    Computation*, 18(8): 1868-1895. (2006).


:api:`mdp.utils.refcast` (array, dtype)
    Cast the array to 'dtype' only if necessary,
    otherwise return a reference.

:api:`mdp.utils.rotate` (mat, angle, columns, units)
    Rotate in-place a NxM data matrix in the plane defined by the 'columns'
    when observation are stored on rows. Observations are rotated
    counterclockwise. This corresponds to the following matrix-multiplication
    for each data-point (unchanged elements omitted)::
 
         [  cos(angle) -sin(angle)     [ x_i ]
            sin(angle)  cos(angle) ] * [ x_j ] 

:api:`mdp.utils.random_rot` (dim, dtype)
    Return a random rotation matrix, drawn from the Haar distribution
    (the only uniform distribution on SO(n)).
    The algorithm is described in the paper
    Stewart, G.W., *The efficient generation of random orthogonal
    matrices with an application to condition estimators*, SIAM Journal
    on Numerical Analysis, 17(3), pp. 403-409, 1980.
    For more information see this `Wikipedia entry
    <http://en.wikipedia.org/wiki/Orthogonal_matrix#Randomization>`_.

:api:`mdp.utils.symrand` (dim_or_eigv, dtype)
    Return a random symmetric (Hermitian) matrix with eigenvalues
    uniformly distributed on (0,1].

HTML Slideshows
---------------

The ``mdp.utils`` module contains some classes and helper function to
display animated results in a Webbrowser. This works by creating an
HTML file with embedded JavaScript code, which dynamically loads
image files (the images contain the content that you want to animate
and can for example be created with matplotlib).
MDP internally uses the open source Templete templating libray,
written by David Bau.

The easiest way to create a slideshow it to use one of these two helper
function:
    
:api:`mdp.utils.show_image_slideshow` (filenames, image_size, filename=None, title=None, \*\*kwargs)
    Write the slideshow into a HTML file, open it in the browser and
    return the file name. ``filenames`` is a list of the images files
    that you want to display in the slideshow. ``image_size`` is a
    2-tuple containing the width and height at which the images should
    be displayed. There are also a couple of additional arguments,
    which are documented in the docstring.
    
:api:`mdp.utils.image_slideshow` (filenames, image_size, title=None, \*\*kwargs)
    This function is similar to ``show_image_slideshow``, but it simply
    returns the slideshow HTML code (including the JavaScript code)
    which you can then embed into your own HTML file. Note that
    the default slideshow CSS code is not included, but it can be
    accessed in ``mdp.utils.IMAGE_SLIDESHOW_STYLE``.
    
Note that there are also two demos for slideshows in the ``mdp\demo``
folder.

Graph module
------------

MDP contains ``mdp.graph``, a lightweight package to handle directed graphs.

:api:`mdp.graph.Graph`

    Represent a directed graph. This class contains several methods
    to create graph structures and manipulate them, among which
    
    - ``add_tree``: Add a tree to the graph.
        The tree is specified with a nested list of tuple, in a LISP-like
        notation. The values specified in the list become the values of
        the single nodes.
        Return an equivalent nested list with the nodes instead of the values.

        Example::
 
            >>> a=b=c=d=e=None
            >>> g.add_tree( (a, b, (c, d ,e)) )
            # corresponds to this tree structure, with all node values set to None:

                    a
                   / \
                  b   c
                     / \
                    d   e

    - ``topological_sort``: Perform a topological sort of the nodes.

    - ``dfs``, ``undirected_dfs``: Perform Depth First sort.

    - ``bfs``, ``undirected_bfs``: Perform Breadth First sort.

    - ``connected_components``: Return a list of lists containing
        the nodes of all connected components of the graph.
    
    - ``is_weakly_connected``: Return True if the graph is weakly connected.

:api:`mdp.graph.GraphEdge`

    Represent a graph edge and all information attached to it.

:api:`mdp.graph.GraphNode`
    Represent a graph node and all information attached to it.

:api:`mdp.graph.recursive_map` (fun, seq)
    Apply a function recursively on a sequence and all subsequences.

:api:`mdp.graph.recursive_reduce` (func, seq, \*argv)
    Apply ``reduce(func, seq)`` recursively to a sequence and all its
    subsequences.