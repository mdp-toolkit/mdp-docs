.. _install:

.. |gE| unicode:: U+2265

************
Installation
************

You can install using :ref:`pip`, :ref:`binary packages <binary_packages>` or 
from :ref:`source <install_from_source>`.

.. _pip:

pip (Windows, MacOSX, Linux)
============================

MDP is listed in the `Python Package Index <http://pypi.python.org/pypi/MDP>`_ and can be
installed with `pip`::

    pip install MDP

This is the preferred method of installation if you are using Windows or MacOSX.

.. _binary_packages:

Binary packages (Linux/BSD)
===========================

.. _python-mdp: http://packages.debian.org/python-mdp

.. _`sci-mathematics/mdp`:
   http://git.overlays.gentoo.org/gitweb/?p=proj/sci.git;a=tree;f=sci-mathematics/mdp

Debian, Ubuntu and derivatives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Thanks to Yaroslav Halchenko, users of Debian, Ubuntu and derivatives can
install the `python-mdp`_ package.

Just type::

    sudo aptitude install python-mdp

Gentoo
~~~~~~
Gentoo users can install the ebuild `sci-mathematics/mdp`_ from the
``science`` overlay.

Use your favourite package manager or, alternatively::

    emerge layman
    layman -L
    layman -a science
    emerge sci-mathematics/mdp

NetBSD
~~~~~~
Thanks to Kamel Ibn Aziz Derouiche, NetBSD users can install the 
`py-mdp <http://pkgsrc.se/wip/py-mdp>`_ from `pkgsrc <http://pkgsrc.se>`_.


.. _install_from_source:

Installation from source
========================

Requirements
~~~~~~~~~~~~
* `Python <http://www.python.org/>`_ 2.6/2.7/3.2/3.3/3.4/3.5
* `Python-Future <http://python-future.org/>`_
* `NumPy <http://numpy.scipy.org/>`_ |gE| 1.6

Download the latest MDP release source archive `here <http://sourceforge.net/projects/mdp-toolkit>`_.

If you want to live on the bleeding edge, check out the MDP git repositories.
You can either `browse the repository <https://github.com/mdp-toolkit/mdp-toolkit>`_
or clone the ``mdp-toolkit`` repository with::

    git clone git://github.com/mdp-toolkit/mdp-toolkit

and then install as explained below.


Installation
~~~~~~~~~~~~
Unpack the archive file and change to the project directory or change to the
cloned git repository, and type::

    python setup.py install

If you want to use MDP without installing it on the system Python path::

    python setup.py install --prefix=/some_dir_in_PYTHONPATH/



Optional Libraries
==================
MDP can make use of several additional libraries if they are installed on your
system. They are not required for using MDP, but may give more
functionality. Here a list of optional libraries and the corresponding
additional features in MDP:

* `SciPy <http://www.scipy.org/>`_ |gE| 0.5.2: Use the fast and
  efficient LAPACK wrapper for the symmetrical eigensolver, used
  interally by many nodes; use the fast FFT routines in some nodes;
  provide the ``Convolution2DNode``, using the fast convolution routines
  in SciPy.
* `Parallel Python <http://www.parallelpython.com/>`_:  provide the
  parallel python scheduler ``PPScheduler`` in the ``parallel``
  module.
* `LibSVM <http://www.csie.ntu.edu.tw/~cjlin/libsvm/>`_ |gE| 2.91:
  provide the ``LibSVMClassifier`` node.
* `joblib <http://packages.python.org/joblib/>`_ |gE| 0.4.3: provide the
  ``caching`` extension and the corresponding ``cache`` context
  manager.
* `sklearn <http://scikit-learn.org/stable/>`_ |gE| 0.6: provide
  wrapper nodes to several sklearn algorithms.

Testing
=======
If you have successfully installed MDP, you can test your installation in a
Python shell as follows::

    >>> import mdp
    >>> mdp.test()
    >>> import bimdp
    >>> bimdp.test()

Note that you will need to install `pytest <http://pytest.org>`_ to run the tests.

If some test fails, please report it to the `mailing list
<https://mail.python.org/mm3/mailman3/lists/mdp-toolkit.python.org/>`_.

.. include:: license.rst
