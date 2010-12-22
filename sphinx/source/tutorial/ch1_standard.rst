**************
Standard Usage
**************

Nodes
=====

A *node* is the basic building block of an MDP application.  It
represents a data processing element, for example a learning
algorithm, a data filter, or a visualization step (see the :ref:`node-list` 
section for an exhaustive list and references).

Each node can have one or more *training phases*, during which the
internal structures are learned from training data (e.g. the weights
of a neural network are adapted or the covariance matrix is estimated)
and an *execution phase*, where new data can be processed forwards (by
processing the data through the node) or backwards (by applying the
inverse of the transformation computed by the node if defined).

Nodes have been designed to be applied to arbitrarily long sets of data;
provided the underlying algorithms support it, the internal structures can
be updated incrementally by sending multiple batches of data (this is
equivalent to online learning if the chunks consists of single
observations, or to batch learning if the whole data is sent in a
single chunk). This makes it possible to perform computations on large amounts
of data that would not fit into memory and to generate data on-the-fly.

A ``Node`` also defines some utility methods, for example
``copy``, which returns an exact copy of a node,  and ``save``, which writes it
in a file. Additional methods may also be present, depending on the
algorithm.

 
Node Instantiation
------------------

A node can be obtained by creating an instance of the ``Node`` class.

Each node is characterized by an input dimension (i.e., the
dimensionality of the input vectors), an output dimension, and a
``dtype``, which determines the numerical type of the internal
structures and of the output signal. By default, these attributes are
inherited from the input data if left unspecified. The constructor of
each node class can require other task-specific arguments. The full
documentation is always available in the doc-string of the node's
class.

Some examples of node instantiation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a node that performs Principal Component Analysis (PCA) 
whose input dimension and ``dtype``
are inherited from the input data during training. Output dimensions
default to input dimensions.
::

    >>> pcanode1 = mdp.nodes.PCANode()
    >>> pcanode1
    PCANode(input_dim=None, output_dim=None, dtype=None)
      
Setting ``output_dim = 10`` means that the node will keep only the 
first 10 principal components of the input.
::

    >>> pcanode2 = mdp.nodes.PCANode(output_dim=10)
    >>> pcanode2
    PCANode(input_dim=None, output_dim=10, dtype=None)

The output dimensionality can also be specified in terms of the explained
variance. If we want to keep the number of principal components which can 
account for 80% of the input variance, we set::

    >>> pcanode3 = mdp.nodes.PCANode(output_dim=0.8)
    >>> pcanode3.desired_variance
    0.80000000000000004

If ``dtype`` is set to ``float32`` (32-bit float), the input 
data is cast to single precision when received and the internal 
structures are also stored as ``float32``. ``dtype`` influences the 
memory space necessary for a node and the precision with which the 
computations are performed.
::

    >>> pcanode4 = mdp.nodes.PCANode(dtype='float32')
    >>> pcanode4
    PCANode(input_dim=None, output_dim=None, dtype='float32')

You can obtain a list of the numerical types supported by a node
looking at its ``supported_dtypes`` property::

    >>> pcanode4.supported_dtypes
    [dtype('float32'), dtype('float64')]

This attribute is a list of ``numpy.dtype`` objects.


A ``PolynomialExpansionNode`` expands its input in the space
of polynomials of a given degree by computing all monomials up
to the specified degree. Its constructor needs as first argument
the degree of the polynomials space (3 in this case).
::

    >>> expnode = mdp.nodes.PolynomialExpansionNode(3)

Node Training
-------------

Some nodes need to be trained to perform their task. For example, the
Principal Component Analysis (PCA) algorithm requires the computation
of the mean and covariance matrix of a set of training data from which
the principal eigenvectors of the data distribution are estimated.

This can be done during a training phases by calling the ``train``
method.  MDP supports both supervised and unsupervised training, and
algorithms with multiple training phases.

Some examples of node training:

Create some random data to train the node::

   >>> x = mdp.numx_rand.random((100, 25))  # 25 variables, 100 observations

Analyzes the batch of data ``x`` and update the estimation of 
mean and covariance matrix::

    >>> pcanode1.train(x)

At this point the input dimension and the ``dtype`` have been
inherited from ``x``::

    >>> pcanode1
    PCANode(input_dim=25, output_dim=None, dtype='float64')

We can train our node with more than one chunk of data. This
is especially useful when the input data is too long to
be stored in memory or when it has to be created on-the-fly.
(See also the :ref:`iterables` section)::

    >>> for i in range(100):
    ...     x = mdp.numx_rand.random((100, 25))
    ...     pcanode1.train(x)
    >>>

Some nodes don't need to or cannot be trained::

    >>> expnode.is_trainable()
    False
  
Trying to train them anyway would raise 
an ``IsNotTrainableException``.

The training phase ends when the ``stop_training``, ``execute``,
``inverse``, and possibly some other node-specific methods are called.
For example we can finalize the PCA algorithm by computing and selecting
the principal eigenvectors::

    >>> pcanode1.stop_training()

If the ``PCANode`` was declared to have a number of output components 
dependent on the input variance to be explained, we can check after
training the number of output components and the actually explained variance::

    >>> pcanode3.train(x)
    >>> pcanode3.stop_training()
    >>> pcanode3.output_dim
    16
    >>> pcanode3.explained_variance
    0.85261144755506446 

It is now possible to access the trained internal data. In general,
a list of the interesting internal attributes can be found in the
class documentation.
::

    >>> avg = pcanode1.avg            # mean of the input data
    >>> v = pcanode1.get_projmatrix() # projection matrix

.. break here

Some nodes, namely the one corresponding to supervised algorithms, e.g.
Fisher Discriminant Analysis (FDA), may need some labels or other
supervised signals to be passed
during training. Detailed information about the signature of the 
``train`` method can be read in its doc-string.
::

    >>> fdanode = mdp.nodes.FDANode()
    >>> for label in ['a', 'b', 'c']:
    ...     x = mdp.numx_rand.random((100, 25))
    ...     fdanode.train(x, label)
    >>> 
      
A node could also require multiple training phases. For example, the
training of ``fdanode`` is not complete yet, since it has two
training phases: The first one computing the mean of the data
conditioned on the labels, and the second one computing the overall
and within-class covariance matrices and solving the FDA
problem. The first phase must be stopped and the second one trained::

    >>> fdanode.stop_training()
    >>> for label in ['a', 'b', 'c']:
    ...     x = mdp.numx_rand.random((100, 25))
    ...     fdanode.train(x, label)
    >>>

The easiest way to train multiple phase nodes is using flows,
which automatically handle multiple phases (see the `Flows`_ section).


Node Execution
--------------

Once the training is finished, it is possible to execute the node:

The input data is projected on the principal components learned
in the training phase::

    >>> x = mdp.numx_rand.random((100, 25))
    >>> y_pca = pcanode1.execute(x)

Calling a node instance is equivalent to executing it::

    >>> y_pca = pcanode1(x)

The input data is expanded in the space of polynomials of
degree 3::

    >>> x = mdp.numx_rand.random((100, 5))
    >>> y_exp = expnode(x)

The input data is projected to the directions learned by FDA::

    >>> x = mdp.numx_rand.random((100, 25))
    >>> y_fda = fdanode(x)

Some nodes may allow for optional arguments in the ``execute`` method. 
As always the complete information can be found in the doc-string.

Node Inversion
-------------- 

If the operation computed by the node is invertible, the node can also
be executed *backwards*, thus computing the inverse transformation:

In the case of PCA, for example, this corresponds to projecting a
vector in the principal components space back to the original data
space::

    >>> pcanode1.is_invertible()
    True
    >>> x = pcanode1.inverse(y_pca)


The expansion node in not invertible::

    >>> expnode.is_invertible()
    False
  
Trying to compute the inverse would raise an ``IsNotInvertibleException``.


.. _write-your-own-nodes:

Writing your own nodes: subclassing ``Node``
--------------------------------------------

MDP tries to make it easy to write new nodes that interface with the
existing data processing elements. 

The ``Node`` class is designed to make the implementation of new
algorithms easy and intuitive. This base class takes care of setting
input and output dimension and casting the data to match the numerical
type (e.g. ``float`` or ``double``) of the internal variables, and offers
utility methods that can be used by the developer.

To expand the MDP library of implemented nodes with user-made nodes,
it is sufficient to subclass ``Node``, overriding some of
the methods according to the algorithm one wants to implement,
typically the ``_train``, ``_stop_training``, and ``_execute``
methods.

In its namespace MDP offers references to the main modules ``numpy``
or ``scipy``, and the subpackages ``linalg``, ``random``, and ``fft``
as ``mdp.numx``, ``mdp.numx_linalg``, ``mdp.numx_rand``, and
``mdp.numx_fft``. This is done to possibly support additional
numerical extensions in the future. For this reason it is recommended
to refer to the ``numpy`` or ``scipy`` numerical extensions through
the MDP aliases ``mdp.numx``, ``mdp.numx_linalg``, ``mdp.numx_fft``,
and ``mdp.numx_rand`` when writing ``Node`` subclasses. This shall
ensure that your nodes can be used without modifications should MDP
support alternative numerical extensions in the future.

We'll illustrate all this with some toy examples.

We start by defining a node that multiplies its input by 2.
  
Define the class as a subclass of ``Node``::
  
    >>> class TimesTwoNode(mdp.Node):

This node cannot be trained. To specify this, one has to overwrite
the ``is_trainable`` method to return False::
  
    ...     def is_trainable(self): 
    ...         return False
  
Execute only needs to multiply ``x`` by 2::

    ...     def _execute(self, x):
    ...         return 2*x

Note that the ``execute`` method, which should never be overwritten
and which is inherited from the ``Node`` parent class, will perform
some tests, for example to make sure that ``x`` has the right rank,
dimensionality and casts it to have the right ``dtype``.  After that
the user-supplied ``_execute`` method is called.  Each subclass has
to handle the ``dtype`` defined by the user or inherited by the
input data, and make sure that internal structures are stored
consistently. To help with this the ``Node`` base class has a method
called ``_refcast(array)`` that casts the input ``array`` only when its
``dtype`` is different from the ``Node`` instance's ``dtype``.

The inverse of the multiplication by 2 is of course the division by 2::
  
    ...     def _inverse(self, y):
    ...         return y/2

Test the new node::

    >>> node = TimesTwoNode(dtype = 'int32')
    >>> x = mdp.numx.array([[1.0, 2.0, 3.0]])
    >>> y = node(x)
    >>> print x, '* 2 =  ', y
    [ [ 1.  2.  3.]] * 2 =   [ [2 4 6]]
    >>> print y, '/ 2 =', node.inverse(y)
    [ [2 4 6]] / 2 = [ [1 2 3]]

We then define a node that raises the input to the power specified
in the initialiser::

    >>> class PowerNode(mdp.Node):

We redefine the init method to take the power as first argument.
In general one should always give the possibility to set the ``dtype``
and the input dimensions. The default value is ``None``, which means that
the exact value is going to be inherited from the input data::

    ...     def __init__(self, power, input_dim=None, dtype=None):
  
Initialize the parent class::

    ...         super(PowerNode, self).__init__(input_dim=input_dim, dtype=dtype)

Store the power::

    ...         self.power = power

``PowerNode`` is not trainable ::

    ...     def is_trainable(self): 
    ...         return False

... nor invertible::

    ...     def is_invertible(self): 
    ...         return False

It is possible to overwrite the function ``_get_supported_dtypes``
to return a list of ``dtype`` supported by the node::

    ...     def _get_supported_dtypes(self):
    ...         return ['float32', 'float64']

The supported types can be specified in any format allowed by the
``numpy.dtype`` constructor. The interface method ``get_supported_dtypes``
converts them and sets the property ``supported_dtypes``, which is
a list of ``numpy.dtype`` objects.

The ``_execute`` method::

    ...     def _execute(self, x):
    ...         return self._refcast(x**self.power)
 
Test the new node::

    >>> node = PowerNode(3)
    >>> x = mdp.numx.array([[1.0, 2.0, 3.0]])
    >>> y = node(x)
    >>> print x, '**', node.power, '=', node(x)
    [ [ 1.  2.  3.]] ** 3 = [ [  1.   8.  27.]]

We now define a node that needs to be trained. The ``MeanFreeNode``
computes the mean of its training data and subtracts it from the input
during execution::

    >>> class MeanFreeNode(mdp.Node):
    ...     def __init__(self, input_dim=None, dtype=None):
    ...         super(MeanFreeNode, self).__init__(input_dim=input_dim, 
    ...                                            dtype=dtype)

We store the mean of the input data in an attribute. We initialize it
to ``None`` since we still don't know how large is an input vector::

    ...         self.avg = None

Same for the number of training points::

    ...         self.tlen = 0
    
The subclass only needs to overwrite the ``_train`` method, which
will be called by the parent ``train`` after some testing and casting has
been done::

    ...     def _train(self, x):
    ...         # Initialize the mean vector with the right 
    ...         # size and dtype if necessary:
    ...         if self.avg is None:
    ...             self.avg = mdp.numx.zeros(self.input_dim,
    ...                                       dtype=self.dtype)
         
Update the mean with the sum of the new data::

    ...         self.avg += mdp.numx.sum(x, axis=0)
 
Count the number of points processed::

    ...         self.tlen += x.shape[0]

Note that the ``train`` method can have further arguments, which might be
useful to implement algorithms that require supervised learning.
For example, if you want to define a node that performs some form
of classification you can define a ``_train(self, data, labels)``
method. The parent ``train`` checks ``data`` and takes care to pass
the ``labels`` on (cf. for example ``mdp.nodes.FDANode``).

The ``_stop_training`` function is called by the parent ``stop_training`` 
method when the training phase is over. We divide the sum of the training 
data by the number of training vectors to obtain the mean::

    ...     def _stop_training(self):
    ...         self.avg /= self.tlen
    ...         if self.output_dim is None:
    ...             self.output_dim = self.input_dim

Note that we ``input_dim`` are set automatically by the ``train`` method,
and we want to ensure that the node has ``output_dim`` set after training.
For nodes that do not need training, the setting is performed automatically
upon execution. The ``_execute`` and ``_inverse`` methods::

    ...     def _execute(self, x):
    ...         return x - self.avg
    ...     def _inverse(self, y):
    ...         return y + self.avg

Test the new node::

    >>> node = MeanFreeNode()
    >>> x = mdp.numx_rand.random((10,4))
    >>> node.train(x)
    >>> y = node(x)
    >>> print 'Mean of y (should be zero):\n', mdp.numx.mean(y, 0)
    Mean of y (should be zero):
    [  0.00000000e+00   2.22044605e-17
      -2.22044605e-17   1.11022302e-17]

It is also possible to define nodes with multiple training phases.
In such a case, calling the ``train`` and ``stop_training`` functions
multiple times is going to execute successive training phases
(this kind of node is much easier to train using Flows_).
Here we'll define a node that returns a meanfree, unit variance signal.
We define two training phases: first we compute the mean of the
signal and next we sum the squared, meanfree input to compute
the standard deviation  (of course it is possible to solve this
problem in one single step - remember this is just a toy example).
::

    >>> class UnitVarianceNode(mdp.Node):
    ...     def __init__(self, input_dim=None, dtype=None):
    ...         super(UnitVarianceNode, self).__init__(input_dim=input_dim, 
    ...                                                dtype=dtype)
    ...         self.avg = None # average
    ...         self.std = None # standard deviation
    ...         self.tlen = 0

The training sequence is defined by the user-supplied method
``_get_train_seq``, that returns a list of tuples, one for each
training phase. The tuples contain references to the training
and stop-training methods of each of them. The default output
of this method is ``[(_train, _stop_training)]``, which explains
the standard behavior illustrated above. We overwrite the method to
return the list of our training/stop_training methods::

    ...     def _get_train_seq(self):
    ...         return [(self._train_mean, self._stop_mean),
    ...                 (self._train_std, self._stop_std)]

Next we define the training methods. The first phase is identical
to the one in the previous example::

    ...     def _train_mean(self, x):
    ...         if self.avg is None:
    ...             self.avg = mdp.numx.zeros(self.input_dim,
    ...                                       dtype=self.dtype)
    ...         self.avg += mdp.numx.sum(x, 0)
    ...         self.tlen += x.shape[0]
    ...     def _stop_mean(self):
    ...         self.avg /= self.tlen

The second one is only marginally different and does not require many
explanations::

    ...     def _train_std(self, x):
    ...         if self.std is None:
    ...             self.tlen = 0
    ...             self.std = mdp.numx.zeros(self.input_dim,
    ...                                       dtype=self.dtype)
    ...         self.std += mdp.numx.sum((x - self.avg)**2., 0)
    ...         self.tlen += x.shape[0]
    ...     def _stop_std(self):
    ...         # compute the standard deviation
    ...         self.std = mdp.numx.sqrt(self.std/(self.tlen-1))

The ``_execute`` and ``_inverse`` methods are not surprising, either::

    ...     def _execute(self, x):
    ...         return (x - self.avg)/self.std
    ...     def _inverse(self, y):
    ...         return y*self.std + self.avg

Test the new node::

    >>> node = UnitVarianceNode()
    >>> x = mdp.numx_rand.random((10,4))
    >>> # loop over phases
    ... for phase in range(2):
    ...     node.train(x)
    ...     node.stop_training()
    ...
    ...
    >>> # execute
    ... y = node(x)
    >>> print 'Standard deviation of y (should be one): ', mdp.numx.std(y, axis=0)
    Standard deviation of y (should be one):  [ 1.  1.  1.  1.]
    

In our last example we'll define a node that returns two copies of its input.
The output is going to have twice as many dimensions.
::

    >>> class TwiceNode(mdp.Node):
    ...     def is_trainable(self): return False
    ...     def is_invertible(self): return False

When ``Node`` inherits the input dimension, output dimension, and ``dtype``
from the input data, it calls the methods ``set_input_dim``, 
``set_output_dim``, and ``set_dtype``. Those are the setters for
``input_dim``, ``output_dim`` and ``dtype``, which are Python 
`properties <http://www.python.org/2.2/descrintro.html>`_. 
If a subclass needs to change the default behavior, the internal methods
``_set_input_dim``, ``_set_output_dim`` and ``_set_dtype`` can
be overwritten. The property setter will call the internal method after
some basic testing and internal settings. The private methods 
``_set_input_dim``, ``_set_output_dim`` and ``_set_dtype`` are responsible
for setting the private attributes ``_input_dim``, ``_output_dim``,
and ``_dtype`` that contain the actual value.
  
Here we overwrite
``_set_input_dim`` to automatically set the output dimension to be twice the
input one, and ``_set_output_dim`` to raise an exception, since
the output dimension should not be set explicitly.
::

    ...     def _set_input_dim(self, n):
    ...         self._input_dim = n
    ...         self._output_dim = 2*n
    ...     def _set_output_dim(self, n):
    ...         raise mdp.NodeException, "Output dim can not be set explicitly!"

The ``_execute`` method::

    ...     def _execute(self, x):
    ...         return mdp.numx.concatenate((x, x), 1)

Test the new node::

    >>> node = TwiceNode()
    >>> x = mdp.numx.zeros((5,2))
    >>> x
    array([[0, 0],
           [0, 0],
           [0, 0],
           [0, 0],
           [0, 0]])
    >>> node.execute(x)
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]])

Flows
=====

A *flow* is a sequence of nodes that are trained and executed
together to form a more complex algorithm.  Input data is sent to the
first node and is successively processed by the subsequent nodes along
the sequence.

Using a flow as opposed to handling manually a set of nodes has a
clear advantage: The general flow implementation automatizes the
training (including supervised training and multiple training phases),
execution, and inverse execution (if defined) of the whole sequence.

Crash recovery is optionally available: in case of failure the current
state of the flow is saved for later inspection. A subclass of the
basic flow class (``CheckpointFlow``) allows user-supplied checkpoint
functions to be executed at the end of each phase, for example to save
the internal structures of a node for later analysis.
Flow objects are Python containers. Most of the builtin ``list``
methods are available. A ``Flow`` can be saved or copied using the
corresponding ``save`` and ``copy`` methods.


Flow instantiation, training and execution
------------------------------------------

For example, suppose we need to analyze a very
high-dimensional input signal using Independent Component Analysis
(ICA). To reduce the computational load, we would like to reduce the
input dimensionality of the data using PCA. Moreover, we would like to
find the data that produces local maxima in the output of the ICA
components on a new test set (this information could be used
for instance to characterize the ICA filters).

We start by generating some input signal at random (which makes the
example useless, but it's just for illustration...).  Generate 1000
observations of 20 independent source signals::

    >>> inp = mdp.numx_rand.random((1000, 20))

Rescale x to have zero mean and unit variance::

    >>> inp = (inp - mdp.numx.mean(inp, 0))/mdp.numx.std(inp, 0)

We reduce the variance of the last 15 components, so that they are
going to be eliminated by PCA::

    >>> inp[:,5:] /= 10.0

Mix the input signals linearly::

    >>> x = mdp.utils.mult(inp,mdp.numx_rand.random((20, 20)))

``x`` is now the training data for our simulation. In the same way
we also create a test set ``x_test``.
::

    >>> inp_test = mdp.numx_rand.random((1000, 20))
    >>> inp_test = (inp_test - mdp.numx.mean(inp_test, 0))/mdp.numx.std(inp_test, 0)
    >>> inp_test[:,5:] /= 10.0
    >>> x_test = mdp.utils.mult(inp_test, mdp.numx_rand.random((20, 20)))

We could now perform our analysis using only nodes, that's the lengthy way...
  
1. Perform PCA::

    >>> pca = mdp.nodes.PCANode(output_dim=5)
    >>> pca.train(x)
    >>> out1 = pca(x)

2. Perform ICA using CuBICA algorithm::

    >>> ica = mdp.nodes.CuBICANode()
    >>> ica.train(out1)
    >>> out2 = ica(out1)

3. Find the three largest local maxima in the output of the ICA node
when applied to the test data, using a ``HitParadeNode``::

    >>> out1_test = pca(x_test)
    >>> out2_test = ica(out1_test)
    >>> hitnode = mdp.nodes.HitParadeNode(3)
    >>> hitnode.train(out2_test)
    >>> maxima, indices = hitnode.get_maxima()

... or we could use flows, which is the best way::

    >>> flow = mdp.Flow([mdp.nodes.PCANode(output_dim=5), mdp.nodes.CuBICANode()])


Note that flows can be built simply by concatenating nodes::
  
    >>> flow = mdp.nodes.PCANode(output_dim=5) + mdp.nodes.CuBICANode()
      
Train the resulting flow::

    >>> flow.train(x)
  
Now the training phase of PCA and ICA are completed. Next we append
a ``HitParadeNode`` which we want to train on the test data::

    >>> flow.append(mdp.nodes.HitParadeNode(3))
    
As before, new nodes can be appended to an existing flow by adding
them ot it::

    >>> flow += mdp.nodes.HitParadeNode(3)
  
Train the ``HitParadeNode`` on the test data::

    >>> flow.train(x_test)
    >>> maxima, indices = flow[2].get_maxima()

A single call to the ``flow``'s ``train`` method will automatically
take care of training nodes with multiple training phases, if such
nodes are present.  

Just to check that everything works properly, we
can calculate covariance between the generated sources and the output
(should be approximately 1)::

    >>> out = flow.execute(x)
    >>> cov = mdp.numx.amax(abs(mdp.utils.cov2(inp[:,:5], out)), axis=1)
    >>> print cov
    [ 0.98992083  0.99244511  0.99227319  0.99663185  0.9871812 ]

The ``HitParadeNode`` is an analysis node and as such does not
interfere with the data flow.
  
Note that flows can be executed by calling the ``Flow`` instance
directly::
     
   >>> out = flow(x)

Flow inversion
--------------

Flows can be inverted by calling their ``inverse`` method.
In the case where the flow contains non-invertible nodes,
trying to invert it would raise an exception.
In this case, however, all nodes are invertible.
We can reconstruct the mix by inverting the flow::

    >>> rec = flow.inverse(out)

Calculate covariance between input mix and reconstructed mix:
(should be approximately 1)
::

    >>> cov = mdp.numx.amax(abs(mdp.utils.cov2(x/mdp.numx.std(x,axis=0),
    ...                                        rec/mdp.numx.std(rec,axis=0))))
    >>> print cov
    [ 0.99839606  0.99744461  0.99616208  0.99772863  0.99690947  
      0.99864056  0.99734378  0.98722502  0.98118101  0.99407939
      0.99683096  0.99756988  0.99664384  0.99723419  0.9985529 
      0.99829763  0.9982712   0.99721741  0.99682906  0.98858858]

Flows are container type objects
--------------------------------

``Flow`` objects are defined as Python containers, and thus are endowed with
most of the methods of Python lists.

You can loop through a ``Flow``::

    >>> for node in flow:
    ...     print repr(node)
    ...
    PCANode(input_dim=20, output_dim=5, dtype='float64')
    CuBICANode(input_dim=5, output_dim=5, dtype='float64')
    HitParadeNode(input_dim=5, output_dim=5, dtype='float64')
    HitParadeNode(input_dim=5, output_dim=5, dtype='float64')
    >>> 

You can get slices, ``pop``, ``insert``, and ``append`` nodes::

    >>> len(flow)
    4
    >>> print flow[::2]
    [PCANode, HitParadeNode]
    >>> nodetoberemoved = flow.pop(-1)
    >>> nodetoberemoved
    HitParadeNode(input_dim=5, output_dim=5, dtype='float64')
    >>> len(flow)
    3
        
Finally, you can concatenate flows::

    >>> dummyflow = flow[1:].copy()
    >>> longflow = flow + dummyflow
    >>> len(longflow)
    4

The returned flow must always be consistent, i.e. input and
output dimensions of successive nodes always have to match. If 
you try to create an inconsistent flow you'll get an exception.


Crash recovery
--------------

If a node in a flow fails, you'll get a traceback that tells you which
node has failed. You can also switch the crash recovery capability on. If
something goes wrong you'll end up with a pickle dump of the flow, that 
can be later inspected.

To see how it works let's define a bogus node that always throws an 
``Exception`` and put it into a flow::

    >>> class BogusExceptNode(mdp.Node):
    ...    def train(self,x):
    ...        self.bogus_attr = 1
    ...        raise Exception, "Bogus Exception"
    ...    def execute(self,x):
    ...        raise Exception, "Bogus Exception"
    ...
    >>> flow = mdp.Flow([BogusExceptNode()])

Switch on crash recovery::
    
    >>> flow.set_crash_recovery(1)

Attempt to train the flow::

    >>> flow.train(x)
    Traceback (most recent call last):
      File "<stdin>", line 1, in ?
      [...]
    mdp.linear_flows.FlowExceptionCR: 
    ----------------------------------------
    ! Exception in node #0 (BogusExceptNode):
    Node Traceback:
    Traceback (most recent call last):
      [...]
    Exception: Bogus Exception
    ----------------------------------------
    A crash dump is available on: "/tmp/MDPcrash_LmISO_.pic"

You can give a file name to tell the flow where to save the dump::

    >>> flow.set_crash_recovery('/home/myself/mydumps/MDPdump.pic')
