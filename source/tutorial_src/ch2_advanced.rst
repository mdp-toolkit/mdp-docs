*****************
Advanced Features
*****************

Iterables
=========

Python allows user-defined classes to support iteration,
as described in the `Python docs 
<http://docs.python.org/library/stdtypes.html#iterator-types>`_. A class is a 
so called iterable if it defines a method ``__iter__`` that returns an 
iterator instance. An iterable is typically some kind of container or 
collection (e.g. ``list`` and ``tuple`` are iterables).

The iterator instance must have a ``next`` method that returns the next 
element in the iteration. In Python an iterable also has to have an 
``__iter__`` method itself that returns ``self`` instead of a new iterator. 
It is important to understand that an iterator only manages a single iteration. 
After this iteration it is spend and cannot be used for a second iteration 
(it cannot be restarted). An iterable on the other hand can create as many 
iterators as needed and therefore supports multiple iterations. Even though 
both iterables and iterators have an ``__iter__`` method they are 
semantically very different (duck-typing can be misleading in this case).

In the context of MDP this means that an iterator can only be used for a 
single training phase, while iterables also support multiple training phases. 
So if you use a node with multiple training phases and train it in a flow 
make sure that you provide an iterable for this node (otherwise an exception 
will be raised). For nodes with a single training phase you can use 
either an iterable or an iterator.

A convenient implementation of the iterator protocol is provided
by generators:
see `this article <http://linuxgazette.net/100/pramode.html>`_ for an
introduction, and the
`official PEP <http://www.python.org/peps/pep-0255.html>`_ for a
complete description.

Let us define two bogus node classes to be used as examples of nodes::

    >>> class BogusNode(mdp.Node):
    ...     """This node does nothing."""
    ...     def _train(self, x):
    ...         pass
    ...
    >>> class BogusNode2(mdp.Node):
    ...     """This node does nothing. But it's not trainable nor invertible.
    ...     """
    ...     def is_trainable(self): return False
    ...     def is_invertible(self): return False
    ...
    >>>


This generator generates ``blocks`` input blocks to be used as training set.
In this example one block is a 2-dimensional time series. The first variable
is [2,4,6,....,1000] and the second one [0,1,3,5,...,999].
All blocks are equal, this of course would not be the case in a real-life
example.

In this example we use a progress bar to get progress information.
::

    >>> def gen_data(blocks):
    ...     for i in mdp.utils.progressinfo(xrange(blocks)):
    ...         block_x = mdp.numx.atleast_2d(mdp.numx.arange(2,1001,2))
    ...         block_y = mdp.numx.atleast_2d(mdp.numx.arange(1,1001,2))
    ...         # put variables on columns and observations on rows
    ...         block = mdp.numx.transpose(mdp.numx.concatenate([block_x,block_y]))
    ...         yield block
    ...
    >>>

The ``progressinfo`` function is a fully configurable text-mode
progress info box tailored to the command-line die-hards. Have a look
at its doc-string and prepare to be amazed!

Let's define a bogus flow consisting of 2 ``BogusNode``::

    >>> flow = mdp.Flow([BogusNode(),BogusNode()], verbose=1)


Train the first node with 5000 blocks and the second node with 3000 blocks. 
Note that the only allowed argument to ``train`` is a sequence (list or 
tuple) of iterables or iterators. In case you don't want or need to use 
incremental learning and want to do a one-shot training, you can use as 
argument to ``train`` a single array of data:

block-mode training
-------------------
::

    >>> flow.train([gen_data(5000),gen_data(3000)])
    Training node #0 (BogusNode)

    [===================================100%==================================>]  

    Training finished
    Training node #1 (BogusNode)
    [===================================100%==================================>]  

    Training finished
    Close the training phase of the last node

**one-shot training** using one single set of data for both nodes
-----------------------------------------------------------------
::

    >>> flow = BogusNode() + BogusNode()
    >>> block_x = mdp.numx.atleast_2d(mdp.numx.arange(2,1001,2))
    >>> block_y = mdp.numx.atleast_2d(mdp.numx.arange(1,1001,2))
    >>> single_block = mdp.numx.transpose(mdp.numx.concatenate([block_x,block_y]))
    >>> flow.train(single_block)

If your flow contains non-trainable nodes, you must specify a ``None``
for the non-trainable nodes::

    >>> flow = mdp.Flow([BogusNode2(),BogusNode()], verbose=1)
    >>> flow.train([None, gen_data(5000)])
    Training node #0 (BogusNode2)
    Training finished
    Training node #1 (BogusNode)
    [===================================100%==================================>]  

    Training finished
    Close the training phase of the last node


You can use the one-shot training::

    >>> flow = mdp.Flow([BogusNode2(),BogusNode()], verbose=1)
    >>> flow.train(single_block)
    Training node #0 (BogusNode2)
    Training finished
    Training node #1 (BogusNode)
    Training finished
    Close the training phase of the last node

Iterators can always be safely used for execution and inversion, since only a 
single iteration is needed::

    >>> flow = mdp.Flow([BogusNode(),BogusNode()], verbose=1)
    >>> flow.train([gen_data(1), gen_data(1)])
    Training node #0 (BogusNode)
    Training finished
    Training node #1 (BosgusNode)
    [===================================100%==================================>]  

    Training finished
    Close the training phase of the last node
    >>> output = flow.execute(gen_data(1000))
    [===================================100%==================================>]  
    >>> output = flow.inverse(gen_data(1000))
    [===================================100%==================================>]  

Execution and inversion can be done in one-shot mode also. Note that
since training is finished you are not going to get a warning
::

    >>> output = flow(single_block)
    >>> output = flow.inverse(single_block)

If a node requires multiple training phases (e.g., 
``GaussianClassifierNode``), ``Flow`` automatically takes care of using the 
iterable multiple times. In this case generators (and iterators) are not 
allowed, since they are spend after yielding the last data block.

However, it is fairly easy to wrap a generator in a simple iterable if you need to::

    >>> class SimpleIterable(object):
    ...     def __init__(self, blocks):
    ...         self.blocks = blocks
    ...     def __iter__(self):
    ...         # this is a generator
    ...         for i in range(self.blocks):
    ...             yield generate_some_data()
    >>>

Note that if you use random numbers within the generator, you usually
would like to reset the random number generator to produce the
same sequence every time::

    >>> class RandomIterable(object):
    ...     def __init__(self):
    ...         self.state = None
    ...     def __iter__(self):
    ...         if self.state is None:
    ...             self.state = mdp.numx_rand.get_state()
    ...         else:
    ...             mdp.numx_rand.set_state(self.state)
    ...         for i in range(2):
    ...             yield mdp.numx_rand.random((1,4))
    >>> iterable = RandomIterable()
    >>> for x in iterable: 
    ...     print x
    ... 
    [[ 0.99586495  0.53463386  0.6306412   0.09679571]]
    [[ 0.51117469  0.46647448  0.95089738  0.94837122]]
    >>> for x in iterable: 
    ...     print x
    ... 
    [[ 0.99586495  0.53463386  0.6306412   0.09679571]]
    [[ 0.51117469  0.46647448  0.95089738  0.94837122]]


Checkpoints
===========

It can sometimes be useful to execute arbitrary functions at the end
of the training or execution phase, for example to save the internal
structures of a node for later analysis. This can easily be done
by defining a ``CheckpointFlow``. As an example imagine the following 
situation: you want to perform Principal Component Analysis (PCA) on 
your data to reduce the dimensionality. After this you want to expand
the signals into a nonlinear space and then perform Slow Feature 
Analysis to extract slowly varying signals. As the expansion will increase
the number of components, you don't want to run out of memory, but at the same
time you want to keep as much information as possible after the dimensionality
reduction. You could do that by specifying the percentage of
the total input variance that has to be conserved in the dimensionality
reduction. As the number of output components of the PCA node now can become 
as large as the that of the input components, you want to check, after training the 
PCA node, that this number is below a certain threshold. If this is not 
the case you want to abort the execution and maybe start again requesting
less variance to be kept.

Let start defining a generator to be used through the whole example::

    >>> def gen_data(blocks,dims):
    ...     mat = mdp.numx_rand.random((dims,dims))-0.5
    ...     for i in xrange(blocks):
    ...         # put variables on columns and observations on rows
    ...         block = mdp.utils.mult(mdp.numx_rand.random((1000,dims)), mat)
    ...         yield block
    ...
    >>>

Define a ``PCANode`` which reduces dimensionality of the input,
a ``PolynomialExpansionNode`` to expand the signals in the space
of polynomials of degree 2 and a ``SFANode`` to perform SFA::

    >>> pca = mdp.nodes.PCANode(output_dim=0.9)
    >>> exp = mdp.nodes.PolynomialExpansionNode(2)
    >>> sfa = mdp.nodes.SFANode()

As you see we have set the output dimension of the ``PCANode`` to be ``0.9``.
This means that we want to keep at least 90% of the variance of the original signal.
We define a ``PCADimensionExceededException`` that has to be thrown when
the number of output components exceeds a certain threshold::

    >>> class PCADimensionExceededException(Exception):
    ...     """Exception base class for PCA exceeded dimensions case."""
    ...     pass
    ...
    >>>


Then, write a ``CheckpointFunction`` that checks the number of output
dimensions of the ``PCANode`` and aborts if this number is larger than ``max_dim``::

    >>> class CheckPCA(mdp.CheckpointFunction):
    ...     def __init__(self,max_dim):
    ...         self.max_dim = max_dim
    ...     def __call__(self,node):
    ...         node.stop_training()
    ...         act_dim = node.get_output_dim()
    ...         if act_dim > self.max_dim:
    ...             errstr = 'PCA output dimensions exceeded maximum '+\
    ...                      '(%d > %d)'%(act_dim,self.max_dim)
    ...             raise PCADimensionExceededException, errstr
    ...         else:
    ...             print 'PCA output dimensions = %d'%(act_dim)
    ...
    >>>

Define the CheckpointFlow::

    >>> flow = mdp.CheckpointFlow([pca, exp, sfa])

To train it we have to supply 3 generators and 3 checkpoint functions::

    >>> flow.train([gen_data(10, 50), None, gen_data(10, 50)],
    ...            [CheckPCA(10), None, None])
    Traceback (most recent call last):
      File "<stdin>", line 2, in ?
      [...]
    __main__.PCADimensionExceededException: PCA output dimensions exceeded maximum (25 > 10)

The training fails with a ``PCADimensionExceededException``.
If we only had 12 input dimensions instead of 50 we would have passed
the checkpoint::

    >>> flow[0] = mdp.nodes.PCANode(output_dim=0.9) 
    >>> flow.train([gen_data(10, 12), None, gen_data(10, 12)],
    ...            [CheckPCA(10), None, None])
    PCA output dimensions = 6

We could use the built-in ``CheckpoinSaveFunction`` to save the ``SFANode`` 
and analyze the results later ::
    
    >>> pca = mdp.nodes.PCANode(output_dim=0.9)
    >>> exp = mdp.nodes.PolynomialExpansionNode(2)
    >>> sfa = mdp.nodes.SFANode()
    >>> flow = mdp.CheckpointFlow([pca, exp, sfa])
    >>> flow.train([gen_data(10, 12), None, gen_data(10, 12)],
    ...            [CheckPCA(10),
    ...             None, 
    ...             mdp.CheckpointSaveFunction('dummy.pic',
    ...                                        stop_training = 1,
    ...                                        protocol = 0)])
    ...
    PCA output dimensions = 7

We can now reload and analyze the ``SFANode``::

    >>> fl = file('dummy.pic')
    >>> import cPickle
    >>> sfa_reloaded = cPickle.load(fl)
    >>> sfa_reloaded
    SFANode(input_dim=35, output_dim=35, dtype='d')
    
Don't forget to clean the rubbish::

    >>> fl.close()
    >>> import os
    >>> os.remove('dummy.pic')


Node Extensions
===============

.. Note::
    The node extension mechanism is an advanced topic, so you might want to
    skip this section at first. The examples here partly use the ``parallel``
    and ``hinet`` packages, which are explained later in the tutorial.

The node extension mechanism makes it possible to dynamically add methods or
class attributes for specific features to node classes (e.g. for
parallelization the nodes need a ``_fork`` and ``_join`` method). Note that
methods are just a special case of class attributes, the extension mechanism
treats them like any other class attributes.
It is also possible for users to define new extensions
to introduce new functionality for MDP nodes without having to directly modify
any MDP code. The node extension mechanism basically enables some
form of *Aspect-oriented programming* (AOP) to deal with *cross-cutting
concerns* (i.e., you want to add a new aspect to node classes which are
spread all over MDP and possibly your own code). In the AOP terminology any
new methods you introduce contain *advice* and the *pointcut* is effectively
defined by the calling of these methods.

Without the extension mechanism the adding of new aspects to nodes would
be done through inheritance, deriving new node classes that implement
the aspect for the parent node class. This is fine unless one wants to use
multiple aspects, requiring multiple inheritance for every combination of
aspects one wants to use. Therefore this approach does not scale well with
the number of aspects.

The node extension mechanism does not directly depend on inheritance, 
instead it adds the methods or class attributes to the node classes 
dynamically at runtime (like *method injection*). This makes it possible 
to activate extensions just when they are needed, reducing the risk of 
interference between different extensions. One can also use multiple 
extensions at the same time, as long as there is no interference, i.e., 
both extensions do not use any attributes with the same name. 

The node extension mechanism uses a special Metaclass, which allows it to  
define the node extensions as classes derived from nodes (bascially just what
one would do without the extension mechanism).
This keeps the code readable and avoids some problems when using automatic
code checkers (like the background pylint checks in the
Eclipse IDE with PyDev).

In MDP the node extension mechanism is currently used by the ``parallel``
package and for the the HTML representation in the ``hinet`` package,
so the best way to learn more is to look there.
We also use these packages in the following examples.

Using Extensions
----------------

First of all you can get all the available node extensions by calling
the ``get_extensions`` function, or to get just a list of their names use
``get_extensions().keys()``. Be careful not to modify the dict returned
by ``get_extensions``, since this will actually modify the registered
extensions. The currently activated extensions are returned
by ``get_active_extensions``. To activate an extension use
``activate_extension``, e.g. to activate the parallel extension
write:
::

    >>> mdp.activate_extension("parallel")
    >>> # now you can use the added attributes / methods
    >>> mdp.deactivate_extension("parallel")
    >>> # the additional attributes are no longer available

.. Note::
    As a user you will never have to activate the parallel extension yourself,
    this is done automatically by the ``ParallelFlow`` class. The parallel
    package will be explained later, it is used here only as an example.
    
Activating an extension adds the available extensions attributes to the 
supported nodes. MDP also provides a context manager for the 
``with`` statement:
::

    >>> with mdp.extension("parallel"):
    ...     pass
    ...
    >>>

The ``with`` statement ensures that the activated extension is deactivated
after the code block, even if there is an exception. Finally there is also a
function decorator:
::

    >>> @mdp.with_extension("parallel")
    ... def f():
    ...     pass
    ...
    >>>
    
Again this ensures that the extension is deactivated after the function call,
even in the case of an exception.

Writing Extension Nodes
-----------------------

Suppose you have written your own nodes and would like to make them compatible
with a particular extension (e.g. add the required methods).
The first way to do this is by using multiple inheritance to derive from
the base class of this extension and your custom node class. For example
the parallel extension of the SFA node is defined in a class::

    >>> class ParallelSFANode(ParallelExtensionNode, mdp.nodes.SFANode):
    ...     def _fork(self):
    ...         # implement the forking for SFANode
    ...         pass
    ...     def _join(self):
    ...         # implement the joining for SFANode
    ...         pass
    ...
    >>>

Here ``ParallelExtensionNode`` is the base class of the extension. Then 
you define the required methods or attributes just like in a normal 
class. If you want you could even use the new ``ParallelSFANode`` class 
like a normal class, ignoring the extension mechanism. Note that your 
extension node is automatically registered in the extension mechanism 
(through a little metaclass magic). 

For methods you can alternatively use the ``extension_method`` function
decorator. You define the extension method like a normal function, but add
the function decorator on top. For example to define the ``_fork`` method
for the ``SFANode`` we could have also used::

    >>> @mdp.extension_method("parallel", mdp.nodes.SFANode) 
    ... def _fork(self):
    ...     pass
    ...
    >>>
        
The first decorator argument is the name of the extension, the second is the
class you want to extend. You can also specify the method name as a third
argument, then the name of the function is ignored (this allows you to get
rid of warnings about multiple functions with the same name).

Creating Extensions
-------------------

To create a new node extension you just have to create a new extension base
class. For example the HTML representation extension in ``mdp.hinet``
is created with::

    >>> class  HTMLExtensionNode(mdp.ExtensionNode, mdp.Node):
    ...     """Extension node for HTML representations of individual nodes."""
    ...     extension_name = "html"
    ...     def html_representation(self):
    ...         pass
    ...     def _html_representation(self):
    ...         pass
    ...
    >>>
            
Note that you must derive from ``ExtensionNode``. If you also derive 
from ``mdp.Node`` then the methods (and attributes) in this class are 
the default implementation for the ``mdp.Node`` class. So they will be 
used by all nodes without a more specific implementation. If you do not 
derive from ``mdp.Node`` then there is no such default implementation. 
You can also derive from a more specific node class if your extension 
only applies to these specific nodes. 

When you define a new extension then you must define the ``extension_name``
attribute. This magic attribute is used to register the new extension and you
can activate or deactivate the extension by using this name.

Note that extensions can override attributes and methods that are 
defined in a node class. The original attributes can still be accessed 
by prefixing the name with ``_non_extension_`` (the prefix string is 
also available as ``mdp.ORIGINAL_ATTR_PREFIX``). On the other hand one 
extension is not allowed to override attributes that were defined by 
another currently active extension.

The extension mechanism uses some magic to make the behavior more 
intuitive with respect to inheritance. Basically methods or attributes 
defined by extensions shadow those which are not defined in the 
extension. Here is an example::

    >>> class TestExtensionNode(mdp.ExtensionNode):
    ...     extension_name = "test"
    ...     def _execute(self):
    ...         return 0
    ...
    >>> class TestNode(mdp.Node):
    ...     def _execute(self):
    ...         return 1
    ...
    >>> class ExtendedTestNode(TestExtensionNode, TestNode):
    ...     pass
    ...
    >>>

After this extension is activated any calls of ``_execute`` in instances 
of ``TestNode`` will return 0 instead of 1. The ``_execute`` from the 
extension base-class shadows the method from ``TestNode``. This makes it 
easier to share behavior for different classes. Without this magic one 
would have to explicitly override ``_execute`` in ``ExtendedTestNode`` 
(or derive the extension base-class from ``Node``, but that would give 
this behavior to all node classes). Note that there is a ``verbose`` 
argument in ``activate_extension`` which can help with debugging. 

Hierarchical Networks
=====================

In case the desired data processing application cannot be defined as a
sequence of nodes, the ``hinet`` subpackage makes it possible to
construct arbitrary feed-forward architectures, and in particular
hierarchical networks.

Building blocks
---------------

The ``hinet`` package contains three basic building blocks (which are all nodes
themselves) to construct hierarchical node networks: ``Layer``, 
``FlowNode``, ``Switchboard``.

The first building block is the ``Layer`` node, which works like a
horizontal version of flow. It acts as a wrapper for a set of nodes
that are trained and executed in parallel. For example, we can
combine two nodes with 100 dimensional input to construct a layer
with a 200-dimensional input
::
      
    >>> node1 = mdp.nodes.PCANode(input_dim=100, output_dim=10)
    >>> node2 = mdp.nodes.SFANode(input_dim=100, output_dim=20)
    >>> layer = mdp.hinet.Layer([node1, node2])
    >>> layer
    Layer(input_dim=200, output_dim=30, dtype=None) 

The first half of the 200 dimensional input data is then
automatically assigned to ``node1`` and the second half to
``node2``. We can train and execute a ``Layer`` just like any other
node. Note that the dimensions of the nodes must be already set when
the layer is constructed.

In order to be able to build arbitrary feed-forward node structures,
``hinet`` provides a wrapper class for flows (i.e., vertical stacks
of nodes) called ``FlowNode``. For example, we can replace
``node1`` in the above example with a ``FlowNode``::

    >>> node1_1 = mdp.nodes.PCANode(input_dim=100, output_dim=50)
    >>> node1_2 = mdp.nodes.SFANode(input_dim=50, output_dim=10)
    >>> node1_flow = mdp.Flow([node1_1, node1_2]) 
    >>> node1 = mdp.hinet.FlowNode(node1_flow)
    >>> layer = mdp.hinet.Layer([node1, node2])
    >>> layer
    Layer(input_dim=200, output_dim=30, dtype=None) 

in this example ``node1`` has two training phases (one for each 
internal node). Therefore ``layer`` now has two training phases as well and 
behaves like any other node with two training phases. 
By combining and nesting ``FlowNode`` and ``Layer``, it is thus possible
to build complex node structures.
 
When implementing networks one might have to route
different parts of the data to different nodes in a layer in complex
ways. This is done by the ``Switchboard`` node, which can handle such
the routing. A ``Switchboard`` is initialized with a 1-D Array with
one entry for each output connection, containing the corresponding
index of the input connection that it receives its input from, e.g.::

    >>> switchboard = mdp.hinet.Switchboard(input_dim=6, connections=[0,1,2,3,4,3,4,5])
    >>> switchboard
    Switchboard(input_dim=3, output_dim=2, dtype=None)
    >>> x = mdp.numx.array([[2,4,6,8,10,12]]) 
    >>> switchboard.execute(x)
    array([[ 2,  4,  6,  8, 10,  8, 10, 12]])

The switchboard can then be followed by a layer that
splits the routed input to the appropriate nodes, as
illustrated in following picture:

.. image:: hinet_switchboard.png
        :width: 400
        :alt: switchboard example

By combining layers with switchboards one can realize any
feed-forward network topology.  Defining the switchboard routing
manually can be quite tedious. One way to automatize this is by
defining switchboard subclasses for special routing situations. The
``Rectangular2dSwitchboard`` class is one such example and will be
briefly described in a later example.

HTML representation
-------------------

Since hierarchical networks can be quite complicated, ``hinet``
includes the class ``HiNetHTMLTranslator`` that translates
an MDP flow into a graphical visualization in an HTML file. We also provide
the helper function ``show_flow`` which creates a complete HTML file with
the flow visualization in it and opens it in your standard browser.
::

    >>> mdp.hinet.show_flow(flow)

To integrate the HTML representation into your own custom HTML file
you can take a look at ``show_flow`` to learn the usage of
``HiNetHTMLTranslator``. You can also specify custom translations for
node types via the extension mechanism (e.g to define which parameters are
displayed).

Example application (2-D image data)
------------------------------------

As promised we now present a more complicated example. We define the
lowest layer for some kind of image processing system. The input
data is assumed to consist of image sequences, with each image having
a size of 50 by 50 pixels. We take color images, so after converting
the images to one dimensional numpy arrays each pixel corresponds to
three numeric values in the array, which the values just next to each
other (one for each color channel).

The processing layer consists of many parallel units, which only see a
small image region with a size of 10 by 10 pixels. These so called
receptive fields cover the whole image and have an overlap of five
pixels. Note that the image data is represented as an 1-D
array. Therefore we need the ``Rectangular2dSwitchboard`` class to
correctly route the data for each receptive field to the corresponding
unit in the following layer.  We also call the switchboard output for
a single receptive field an output channel and the three RGB values
for a single pixel form an input channel.  Each processing unit is a
flow consisting of an ``SFANode`` (to somewhat reduce the
dimensionality) that is followed by an ``SFA2Node``. Since we assume
that the statistics are similar in each receptive filed we actually
use the same nodes for each receptive field. Therefore we use a
``CloneLayer`` instead of the standard ``Layer``. Here is the actual
code::

    >>> switchboard = mdp.hinet.Rectangular2dSwitchboard(x_in_channels=50, 
    ...                                                  y_in_channels=50, 
    ...                                                  x_field_channels=10, 
    ...                                                  y_field_channels=10,
    ...                                                  x_field_spacing=5, 
    ...                                                  y_field_spacing=5,
    ...                                                  in_channel_dim=3)
    >>> sfa_dim = 48
    >>> sfa_node = mdp.nodes.SFANode(input_dim=switchboard.out_channel_dim, 
    ...                              output_dim=sfa_dim)
    >>> sfa2_dim = 32
    >>> sfa2_node = mdp.nodes.SFA2Node(input_dim=sfa_dim, 
    ...                                output_dim=sfa2_dim)
    >>> flownode = mdp.hinet.FlowNode(mdp.Flow([sfa_node, sfa2_node]))
    >>> sfa_layer = mdp.hinet.CloneLayer(flownode, 
    ...                                  n_nodes=switchboard.output_channels)
    >>> flow = mdp.Flow([switchboard, sfa_layer])

The HTML representation of the the constructed flow looks like this in your
browser:

.. image:: hinet_html.png
        :width: 450
        :alt: hinet HTML rendering

Now one can train this flow for example with image sequences from a movie.
After the training phase one can compute the image pattern that produces
the highest response in a given output coordinate 
(use ``mdp.utils.QuadraticForm``). One such optimal image pattern may
look like this (only a grayscale version is shown): 

.. image:: hinet_opt_stim.png
        :alt: optimal stimulus

So the network units have developed some kind of primitive line
detector. More on this topic can be found in: Berkes, P. and Wiskott,
L., `Slow feature analysis yields a rich repertoire of complex cell
properties`.  
`Journal of Vision, 5(6):579-602 <http://journalofvision.org/5/6/9/>`_. 

One could also add more layers on top of this first layer to do more 
complicated stuff. Note that the ``in_channel_dim`` in the next 
``Rectangular2dSwitchboard`` would be 32, since this is the output dimension 
of one unit in the ``CloneLayer`` (instead of 3 in the first switchboard, 
corresponding to the three RGB colors).

Parallelization
===============

The ``parallel`` package adds the ability to parallelize the training 
and execution of MPD flows. This package is split into two decoupled parts:

The first part consists of a parallel extension of the familiar MDP
structures of nodes and flows. The first basic building block is the
extension class ``ParallelExtensionNode`` for nodes which can be trained
in a parallelized way. It adds the ``fork`` and ``join`` methods. When
providing a parallel extension for custom node classes you should provide
``_fork`` and ``_join``.
Secondly there is the ``ParallelFlow`` class,
which internally splits the training or execution into tasks which can 
then be processed in parallel.

The second part consists of the schedulers. A scheduler takes tasks
and processes them in a more or less parallel way (e.g. in multiple
Python processes). A scheduler deals with the more technical aspects
of the parallelization, but does not need to know anything about
nodes and flows.

Basic Examples
--------------
In the following example we parallelize a simple ``Flow`` consisting of
PCA and quadratic SFA, so that it makes use of multiple cores on a modern CPU:
::

    >>> node1 = mdp.nodes.PCANode(input_dim=100, output_dim=10)
    >>> node2 = mdp.nodes.SFA2Node(input_dim=10, output_dim=10)
    >>> parallel_flow = mdp.parallel.ParallelFlow([node1, node2])
    >>> n_data_chunks = 10
    >>> data_iterables = [[np.random.random((50, 100))
    ...                    for _ in range(n_data_chunks)]] * 2
    >>> scheduler = mdp.parallel.ProcessScheduler()
    >>> parallel_flow.train(data_iterables, scheduler=scheduler)
    >>> scheduler.shutdown()

Only two additional lines were needed to parallelize the training of the 
flow. All one has to do is use a ``ParallelFlow`` instead of the normal 
``Flow`` and provide a scheduler. The ``ProcessScheduler`` will 
automatically create as many Python processes as there are CPU cores. 
The parallel flow gives the training task for each data chunk over to 
the scheduler, which in turn then distributes them across the available 
worker processes. The results are then returned to the flow, which puts 
them together in the right way. Note that the ``shutdown`` method should 
be always called at the end to make sure that the recources used by the 
scheduler are cleaned up properly. One should therefore put the 
``shutdown`` call into a safe try/finally statement: 
::

    >>> scheduler = mdp.parallel.ProcessScheduler()
    >>> try:
    ...     parallel_flow.train(data_iterables, scheduler=scheduler)
    ... finally:
    ...     scheduler.shutdown()
    ...
    
The ``Scheduler`` class also supports the context manager interface of Python.
One can therefore use a ``with`` statement:
::

    >>> with mdp.parallel.ProcessScheduler() as scheduler:
    ...     parallel_flow.train(data_iterables, scheduler=scheduler)
    ...
    
The ``with`` statement ensures that ``scheduler.shutdown`` is automatically
called (even if there is an exception).
 

Scheduler
---------

The scheduler classes in MDP are derived from the ``Scheduler`` base 
class (which itself does not implement any parallelization). The 
standard choice at the moment is the ``ProcessScheduler``, which 
distributes the incoming tasks over multiple Python processes 
(circumventing the global interpreter lock or GIL). The performance gain 
is highly dependent on the specific situation, but can potentially scale 
well with the number of CPU cores (in one real world case we saw a 
speed-up factor of 4.2 on an Intel Core i7 processor with 4 physical / 8 
logical cores). 

MDP has experimental support for the `Parallel Python library 
<http://www.parallelpython.com>`_ in the ``mdp.parallel.pp_support`` 
package. In principle this makes it possible to parallelize across 
multiple machines. Recently we also added the thread based scheduler 
``ThreadScheduler``. While it is limited by the GIL it can still 
achieve a real-world speedup (since NumPy releases the GIL when 
possible) and it causes less overhead compared to the 
``ProcessScheduler``.

(The following information is only releveant for people who want to implement
custom scheduler classes.)

The first important method of the scheduler class is ``add_task``. This 
method takes two arguments: ``data`` and ``task_callable``, which can be 
a function or an object with a ``__call__`` method. The return value of 
the ``task_callable`` is the result of the task. If ``task_callable`` is 
``None`` then the last provided ``task_callable`` will be used. This 
splitting into callable and data makes it possible to implement caching 
of the ``task_callable`` in the scheduler and its workers (caching is 
turned on by default in the ``ProcessScheduler``). To further influence 
caching you can also derive from the ``TaskCallable`` class, which has a 
``fork`` method to generate new callables when the cached callable must 
be preserved. For MDP training and execution there are corresponding 
classes derived from ``TaskCallable`` which are automatically used, so 
normally there is no need to worry about this. 

After submitting all the tasks with ``add_task`` you can then call
the ``get_results`` method. This method returns all the task results,
normally in a list. If there are open tasks in the scheduler then
``get_results`` will wait until all the tasks are finished (it blocks). You can
also check the status of the scheduler by looking at the
``n_open_tasks`` property, which gives you the number of open tasks.
After using the scheduler you should always call the ``shutdown`` method,
otherwise you might get error messages from not properly closed processes.

Internally an instance of the base class ``mdp.parallel.ResultContainer`` is
used for the storage of the results in the scheduler. By providing your own
result container to the scheduler you modify the storage. For example the
default result container is an instance of ``OrderedResultContainer``. The
``ParallelFlow`` class by default makes sure that the right container is
used for the task (this can be overriden manually via the
``overwrite_result_container`` parameter of the ``train`` and ``execute``
methods).

Parallel Nodes
--------------

If you want to parallelize your own nodes you have to provide parallel
extensions for them. The ``ParallelExtensionNode`` base class has
the new template methods ``fork`` and ``join``. 
``fork`` should return a new node instance. This new instance can then be
trained somewhere else (e.g. in a different process) with the usual ``train``
method. Afterwards one calls ``join`` on the original node, with the
forked node as the argument. This is effectively the same as calling
``train`` directly on the original node.

When writing your own parallel node extension you should only overwrite the 
``_fork`` and ``_join`` methods, which are automatically called by ``fork`` and
``join``. The ``fork`` and ``join`` take care of the standard node
attributes like the dimensions. You should also look at the source
code of a parallel node like ``ParallelPCANode`` to get a better idea
of how to parallelize nodes.

Currently we provide the following parallel nodes:
``ParallelPCANode``, ``ParallelWhiteningNode``, ``ParallelSFANode``,
``ParallelSFA2Node``, ``ParallelFDANode``, ``ParallelHistogramNode``,
``ParallelAdaptiveCutoffNode``, ``ParallelFlowNode``, ``ParallelLayer``,
``ParallelCloneLayer`` (the last three are derived from the ``hinet``
package).


Classifier nodes
================

New in MDP 2.6 is the ``ClassifierNode`` base class which offers a simple
interface for creating classification tasks. Usually, one does not want to use
the classification output in a flow but extract this information independently.
Most classification nodes will therefore simply return the identity function on
``execute``; all classification work is done with the new methods ``label``,
``prob`` and ``rank``.

As a first example, we will use the ``GaussianClassifierNode``.
::

    >>> gc = mdp.nodes.GaussianClassifierNode()
    >>> gc.train(mdp.numx_rand.random((50, 3)), +1)
    >>> gc.train(mdp.numx_rand.random((50, 3)) - 0.8, -1)
	
We have trained the node and assigned the labels +1 and -1 to the sample points.
Note that in this simple case we don’t need to give a label to each individual point,
when only a single label is given, it is assigned to the whole batch of features.
However, it is also possible to use the more explicit form::

    >>> gc.train(mdp.numx_rand.random((50, 3)), [+1] * 50)
	
We can then retrieve the most probable labels for some testing data,
::

    >>> test_data = mdp.numx.array([[0.1, 0.2, 0.1], [-0.1, -0.2, -0.1]])
    >>> gc.label(test_data)
    [1, -1]
	
and also get the probability for each label.
::

    >>> gc.prob(test_data)
    [{-1: 0.21013407927789607, 1: 0.78986592072210393},
     {-1: 0.99911458988539714, 1: 0.00088541011460285866}]


Finally, it is possible to get the ranking of the labels, starting with the likeliest.
::

    >>> gc.rank(test_data)
    [[1, -1], [-1, 1]]
	

New nodes should inherit from ``ClassifierNode`` and implement the ``_label`` and ``_prob``
methods. The public ``rank`` method will be created automatically from ``prob``.