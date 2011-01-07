.. _caching:

=========================
Caching execution results
=========================
.. codesnippet::

Intro
-----

It is relatively common for nodes to process the same data several
times. Usually this happens when training a long sequence of nodes
using a fixed data set: to train the nodes at end of the sequence, the
data has to be processed by all the preceding ones. This duplication
of efforts may be costly, for example in image processing, when one
needs to repeatedly filter the images.

MDP offers a node extension that automatically caches the result of
the ``execute`` method, which can boost the speed of your application
considerably in such scenarios. The cache can be activated globally
(i.e., for all node instances), for a specific node class, or for
specific instances.

The caching mechanism is based on the library 
`joblib <http://packages.python.org/joblib/>`_, version 0.4.3 or higher.

Activating the extension
------------------------

* use activate_cache

* on classes

* on instances

remember to de-activate or the cache may be in a broken state

Cache context manager
---------------------

* how to use
