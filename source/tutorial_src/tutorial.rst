.. _tutorial:

********
Tutorial
********

:Author: Pietro Berkes, Rike-Benjamin Schuppner, Niko Wilbert and Tiziano Zito
:Homepage: http://mdp-toolkit.sourceforge.net
:Copyright: This document has been placed in the public domain.
:Version: 2.6

This document is also available as `pdf file<http://downloads.sourceforge.net/mdp-toolkit/MDP2_6_tutorial.pdf?download>`_.

This is a guide to basic and some more advanced features of
the MDP library. Besides the present tutorial, you can learn 
more about MDP by using the standard Python tools.  
All MDP nodes have doc-strings, the public
attributes and methods have telling names: All information about a 
node can be obtained using  the ``help`` and ``dir`` functions within 
the Python interpreter. In addition to that, an automatically generated 
`API <http://mdp-toolkit.sourceforge.net/docs/api/index.html>`_ is 
available.

.. Note::
  Code snippets throughout the script will be denoted by:

  ::

      >>> print "Hello world!"
      Hello world!

  To run the following code examples don't forget to import mdp
  in your Python session with:
  ::
  
     >>> import mdp

  You'll find all the code of this tutorial within the ``demo`` directory
  in the MDP installation path. 

.. toctree::

   ch1_standard.rst
   ch2_advanced.rst
   ch3_nodesutils.rst
   ch4_bimdp.rst
 
.. include:: <isonum.txt>

Future Development
------------------

MDP is currently maintained by a core team of 4 developers, but it is
open to user contributions. Users have already contributed some of the
nodes, and more contributions are currently being reviewed for
inclusion in future releases of the package. The package development
can be followed online on the public git code
`repositories <http://mdp-toolkit.git.sourceforge.net>`_ or cloned with:
::

    git clone git://mdp-toolkit.git.sourceforge.net/gitroot/mdp-toolkit/mdp-toolkit
    git clone git://mdp-toolkit.git.sourceforge.net/gitroot/mdp-toolkit/docs
    git clone git://mdp-toolkit.git.sourceforge.net/gitroot/mdp-toolkit/examples
    git clone git://mdp-toolkit.git.sourceforge.net/gitroot/mdp-toolkit/contrib

Questions, bug reports, and feature requests are typically handled by
the user `mailing list <https://lists.sourceforge.net/mailman/listinfo/mdp-toolkit-users>`_


Contributors
------------
In this final section we want to thank all users who have contributed
code and bug reports to the MDP project. Strictly in alphabetical order:

- `Gabriel Beckers <http://www.gbeckers.nl/>`_
- Sven Dähne
- Alberto Escalante
- `Farzad Farkhooi <http://www.bccn-berlin.de/People/farkhooi>`_
- Mathias Franzius
- `Michael Hanke <http://apsy.gse.uni-magdeburg.de/main/index.psp?page=hanke/main&lang=en&sec=0>`_
- `Konrad Hinsen <http://dirac.cnrs-orleans.fr/~hinsen/>`_
- Christian Hinze
- `Samuel John <http://www.samueljohn.de/>`_
- Susanne Lezius
- `Michael Schmuker <http://userpage.fu-berlin.de/~schmuker/>`_
- `Benjamin Schrauwen <http://snn.elis.ugent.be/benjamin>`_
- `Henning Sprekeler <http://lcn.epfl.ch/~sprekele>`_
- `Jake VanderPlas <http://www.astro.washington.edu/vanderplas/>`_
