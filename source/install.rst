.. _install:

.. |gE| unicode:: U+2265

************
Installation
************

You can install :ref:`binary packages <binary_packages>` or 
from :ref:`source <install_from_source>`.

.. _binary_packages:

Binary packages (Linux)
=======================

MDP is prepackaged for:

.. _python-mdp: http://packages.debian.org/python-mdp

* `Debian <http://www.debian.org>`_ (package `python-mdp`_)
* `Ubuntu <http://www.ubuntu.com>`_ (package `python-mdp`__)
* `Gentoo <http://www.gentoo.org>`_ (ebuild `sci-mathematics/mdp`_)
* `Mandriva <http://www.mandriva.com/en/>`_ (package `python-mdp`)
* `PCLinuxOS <http://www.pclinuxos.com>`_ (package `python-mdp`)


__ http://packages.ubuntu.com/python-mdp

.. _py25-mdp-toolkit:
   http://trac.macports.org/browser/trunk/dports/python/py25-mdp-toolkit/Portfile
.. _py26-mdp-toolkit:
   http://trac.macports.org/browser/trunk/dports/python/py26-mdp-toolkit/Portfile
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

Binary Packages (MacOSX)
========================
Thanks to Maximilian Nickel, Mac OS X users using `MacPorts
<http://www.macports.org/>`_ can install the packages `py25-mdp-toolkit
<http://trac.macports.org/browser/trunk/dports/python/py25-mdp-toolkit/Portfile>`_
and `py26-mdp-toolkit`_,
respectively.

If you use the `MacPorts <http://www.macports.org/>`_ system, just type::

    sudo port install py25-mdp-toolkit

or::

    sudo port install py26-mdp-toolkit

depending on your favoured version of Python.

If you happen to use `homebrew <http://mxcl.github.com/homebrew/>`_ as
your OS X package manager, you will have to install from the archive
or alternatively, use ``pip`` (see `Python installers`_ below).

Binary Packages (Microsoft Windows)
===================================

On Windows, the installation of the binary distribution is as easy as executing
the installer and following the instructions. Get the installer `here
<http://sourceforge.net/projects/mdp-toolkit>`_. If you run Windows
Vista or Windows 7, you may need to right-click on the installer and
pick *Run as Administrator*.


Python installers
=================

MDP is also listed in the `Python Package Index <http://pypi.python.org/pypi/MDP>`_.

Users who like to install MDP using Python’s own packaging tools may
want to use either:

.. download-link:: pip install

or

.. download-link:: easy_install

If you use ``pip`` you can even install from the git repository
directly with: ::

    pip install -e git://github.com/mdp-toolkit/mdp-toolkit#egg=MDP

.. _install_from_source:

Installation from source
========================

Requirements
~~~~~~~~~~~~
* `Python <http://www.python.org/>`_ 2.5/2.6/2.7/3.1/3.2/3.3
* `NumPy <http://numpy.scipy.org/>`_ |gE| 1.1

Download the latest MDP release source archive `here <http://sourceforge.net/projects/mdp-toolkit>`_.

If you want to live on the bleeding edge, check out the MDP git repositories.
You can either `browse the repository <mdp-toolkit-commits>`
or clone the ``mdp-toolkit`` repository with::

    git clone git://github.com/mdp-toolkit/mdp-toolkit

and then install as explained below.

.. _mdp-toolkit-commits:
    https://github.com/mdp-toolkit/mdp-toolkit/commits/master


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
* `Shogun <http://www.shogun-toolbox.org/>`_ |gE| 1.0: provide the
  ``ShogunSVMClassifier``  node.
* `LibSVM <http://www.csie.ntu.edu.tw/~cjlin/libsvm/>`_ |gE| 2.91:
  provide the ``LibSVMClassifier`` node.
* `joblib <http://packages.python.org/joblib/>`_ |gE| 0.4.3: provide the
  ``caching`` extension and the corresponding ``cache`` context
  manager.
* `sklearn <http://scikit-learn.org/>`_ |gE| 0.6: provide
  wrapper nodes to several sklearn algorithms.


Python 3
========
MDP supports Python 3.X and Python 2.X within a single code base. Thanks
to the great work by Pauli Virtanen and David Cournapeau of the NumPy
developers' team, the Python 3 compatible code is generated
automatically when you install with Python 3. Note that NumPy is
compatible with Python 3 since release 1.5. 

Testing
=======
If you have successfully installed MDP, you can test your installation in a
Python shell as follows::

    >>> import mdp
    >>> mdp.test()
    >>> import bimdp
    >>> bimdp.test()

If some test fails, please report it to the `mailing list
<https://lists.sourceforge.net/lists/listinfo/mdp-toolkit-users>`_.

.. include:: license.rst
