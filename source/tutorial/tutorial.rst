.. _tutorial:

********
Tutorial
********

:Author: Pietro Berkes, Rike-Benjamin Schuppner, Niko Wilbert and Tiziano Zito
:Homepage: http://mdp-toolkit.sourceforge.net
:Copyright: This document has been placed in the public domain.
:Version: 2.6

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
  Code snippets throughout the script will be denoted by
 
      >>> print "Hello world!" #doctest: +SKIP
      Hello world! # doctest: +SKIP

  To run the following code examples don't forget to import ``mdp``
  and ``numpy`` in your Python session with
  
     >>> import mdp
     >>> import numpy as np

  You'll find all the code of this tutorial within the ``demo`` directory
  in the MDP installation path. 

.. toctree::

   ch0_introduction.rst
   ch1_standard.rst
   ch2_advanced.rst
   ch3_nodesutils.rst
   ch4_bimdp.rst
 
