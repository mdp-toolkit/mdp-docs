.. _tutorial:

********
Tutorial
********

.. version-string::


This document is also available as `pdf file`_.

.. _`pdf file`: http://downloads.sourceforge.net/mdp-toolkit/MDP2_6_tutorial.pdf?download

This is a guide to basic and some more advanced features of
the MDP library. Besides the present tutorial, you can learn 
more about MDP by using the standard Python tools.  
All MDP nodes have doc-strings, the public
attributes and methods have telling names: All information about a 
node can be obtained using  the ``help`` and ``dir`` functions within 
the Python interpreter. In addition to that, an automatically generated 
`API documentation`_ is available.

.. _`API documentation`: ../api/index.html

.. Note::
  Code snippets throughout the script will be denoted by::
 
      >>> print "Hello world!"
      Hello world!

  To run the following code examples don't forget to import ``mdp``
  and ``numpy`` in your Python session with::
  
     >>> import mdp
     >>> import numpy as np

  You'll find all the code of this tutorial :ref:`here <code_snippets>`.

.. toctree::

   quick_start.rst
   introduction.rst
   nodes.rst
   flows.rst
   iterables.rst
   checkpoints.rst
   extensions.rst
   hinet.rst
   parallel.rst
   classifiers.rst
   node_list.rst
   additional_utilities.rst
   bimdp.rst
 
