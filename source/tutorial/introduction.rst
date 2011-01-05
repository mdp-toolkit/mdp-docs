.. _introduction:

************
Introduction
************

.. |reg| unicode:: U+00AE
.. |trade| unicode:: U+2122

The use of the Python programming language in computational
neuroscience has been growing steadily over the past few years. The
maturation of two important open source projects, the scientific
libraries `NumPy <http://numpy.scipy.org>`_ and 
`SciPy <http://www.scipy.org>`_, gives access to a large
collection of scientific functions which rival in size and speed those from 
well known commercial alternatives such as `Matlab`\ |reg| from The MathWorks\ |trade|.

Furthermore, the flexible and dynamic nature of Python offers 
scientific programmers the opportunity to quickly develop efficient and
structured software while maximizing prototyping and reusability
capabilities.

The `Modular toolkit for Data Processing (MDP)
<http://mdp-toolkit.sourceforge.net>`_ package contributes to this
growing community a library of widely used data processing algorithms,
and the possibility to combine them together to form pipelines for
building more complex data processing software.

MDP has been designed to be used as-is and as a framework for
scientific data processing development.

From the user's perspective, MDP consists of a collection of *units* 
which process data. For example, these include algorithms for supervised 
and unsupervised learning, principal & independent components analysis 
and classification.

These *units* can be chained into data processing, *flows*, to create pipelines
as well as more complex feed-forward network architectures. Given a set of
input data, MDP takes care of training and executing all nodes in the network
in the correct order and  passing intermediate data between the nodes. This
allows the user to specify complex algorithms as a series of simpler data
processing steps. 

The number of available algorithms is steadily increasing and includes,
to name just the most common, Principal Component Analysis
(:api:`mdp.nodes.PCANode <PCA>` and
:api:`mdp.nodes.NIPALSNode <NIPALS>`),
several Independent Component Analysis algorithms (:api:`mdp.nodes.CuBICANode <CuBICA>`,
:api:`mdp.nodes.FastICANode <FastICA>`,
:api:`mdp.nodes.TDSEPNode <TDSEP>`,
:api:`mdp.nodes.JADENode <JADE>`,
and :api:`mdp.nodes.XSFANode <XSFA>`),
:api:`mdp.nodes.SFANode <Slow Feature Analysis>`,
:api:`mdp.nodes.GaussianClassifierNode <Gaussian Classifiers>`,
:api:`mdp.nodes.RBMNode <Restricted Boltzmann Machine>`,
and :api:`mdp.nodes.LLENode <Locally Linear Embedding>`
(see the :ref:`node_list` section for a more exhaustive list and
references).

Particular care has been taken to make computations efficient in terms of speed
and memory.  To reduce the memory footprint, it is possible to perform learning
using batches of data. For large data-sets, it is also possible to specify that
MDP should use single precision floating point numbers rather than double
precision ones.  Finally, calculations can be parallelised using the
``parallel`` subpackage, which offers a parallel implementation of the basic
nodes and flows.

From the developer's perspective, MDP is a framework that makes the
implementation of new supervised and unsupervised learning algorithms
easy and straightforward.  The basic class, ``Node``, takes care of
tedious tasks like numerical type and dimensionality checking, leaving
the developer free to concentrate on the implementation of the
learning and execution phases. Because of the common interface, the
node then automatically integrates with the rest of the library and
can be used in a network together with other nodes. 

A node can have multiple training phases and even an undetermined number 
of phases. Multiple training phases mean that the training data is 
presented multiple times to the same node. This allows the 
implementation of algorithms that need to collect some statistics on the 
whole input before proceeding with the actual training, and others that 
need to iterate over a training phase until a convergence criterion is 
satisfied. It is possible to train each phase using chunks of input data 
if the chunks are given as an iterable. Moreover, crash recovery can be 
optionally enabled, which will save the state of the flow in case of a 
failure for later inspection.

MDP is distributed under the open source :ref:`BSD license <license>`. It
has been written in the context of theoretical research in
neuroscience, but it has been designed to be helpful in any context
where trainable data processing algorithms are used. Its simplicity on
the user's side, the variety of readily available algorithms, and the
reusability of the implemented nodes also make it a useful educational
tool.

With over 10,000 downloads since its first public release in 2004, MDP
has become a widely used Python scientific software. It has minimal
dependencies, requiring only the NumPy numerical extension, is
completely platform-independent, and is available as a
`package <http://packages.debian.org/python-mdp>`_
in the GNU/Linux 
`Debian <http://www.debian.org>`_ distribution and the
`Python(x,y) <http://www.pythonxy.com>`_ scientific Python
distribution.

As the number of users and contributors is increasing, MDP appears
to be a good candidate for becoming a community-driven common
repository of user-supplied, freely available, Python implemented data
processing algorithms.
