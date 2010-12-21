.. install:

************
Installation
************

Requirements
============

.. |gE| unicode:: U+2265

* `Python <http://www.python.org/>`_ 2.5/2.6/2.7/3.1/3.2
* `NumPy <http://numpy.scipy.org/>`_ |gE| 1.1 

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
* `Shogun <http://www.shogun-toolbox.org/>`_ |gE| 0.9: provide the
  ``ShogunSVMClassifier``  node.
* `LibSVM <http://www.csie.ntu.edu.tw/~cjlin/libsvm/>`_ |gE| 2.91:
  provide the ``LibSVMClassifier`` node.
* `joblib <http://packages.python.org/joblib/>`_: provide the
  ``caching`` extension and the corresponding ``cache`` context
  manager.
* `scikits.learn <http://scikit-learn.sourceforge.net/>`_: provide
  wrapper nodes to several scikits.learn algorithms.

Python 3
========
MDP supports Python 3.X and Python 2.X within a single code base. Thanks
to the great work by Pauli Virtanen and David Cournapeau of the NumPy
developers' team, the Python 3 compatible code is generated
automatically when you install with Python 3. Note that NumPy is
compatible with Python 3 since release 1.5. At the moment there are
still no binary packages of NumPy for Python 3, so you may have to
build NumPy from `source <https://github.com/numpy/numpy>`_.

License
=======
MDP is distributed under the open source `BSD license <http://mdp-toolkit.git.sourceforge.net/git/gitweb.cgi?p=mdp-toolkit/mdp-toolkit;a=blob_plain;f=COPYRIGHT;hb=HEAD>`_. 

Download
========

Download the latest MDP release `here <http://sourceforge.net/projects/mdp-toolkit>`_.


If you want to live on the bleeding edge, check out the MDP git repositories.
You can either `browse the repository <http://mdp-toolkit.git.sourceforge.net/git/gitweb.cgi?p=mdp-toolkit/mdp-toolkit;a=summary>`_ or clone the ``mdp-toolkit``
repository with: ::

    git clone git://mdp-toolkit.git.sourceforge.net/gitroot/mdp-toolkit/mdp-toolkit

and then install as explained below.

Thanks to Yaroslav Halchenko, users of Debian, Ubuntu and derivatives can
install the `python-mdp <http://packages.debian.org/python-mdp>`_
package.

Thanks to Maximilian Nickel, Mac OS X users using `MacPorts
<http://www.macports.org/>`_ can install the packages `py25-mdp-toolkit
<http://trac.macports.org/browser/trunk/dports/python/py25-mdp-toolkit/Portfile>`_
and `py26-mdp-toolkit
<http://trac.macports.org/browser/trunk/dports/python/py26-mdp-toolkit/Portfile>`_,
respectively.

Gentoo users can install the ebuild `sci-mathematics/mdp
<http://git.overlays.gentoo.org/gitweb/?p=proj/sci.git;a=tree;f=sci-mathematics/mdp>`_ from the
``science`` overlay.

Installation
============

Unpack the archive file and change to the project directory or change to the
cloned git repository, and type: ::

    python setup.py install

If you want to use MDP without installing it on the system Python path: ::

    python setup.py install --prefix=/some_dir_in_PYTHONPATH/

On Debian you can just type: ::

    sudo aptitude install python-mdp

On Mac OS X if you use the `MacPorts <http://www.macports.org/>`_ system, just type: ::

    sudo port install py25-mdp-toolkit

or: ::

    sudo port install py26-mdp-toolkit

depending on your favoured version of Python.
On Gentoo you can use your favourite package manager or, alternatively: ::

    emerge layman
    layman -L
    layman -a science
    emerge sci-mathematics/mdp

On Windows, the installation of the binary distribution is as easy as executing
the installer and following the instructions.

Testing
=======

If you have successfully installed MDP, you can test your installation in a
Python shell as follows: ::

    >>> import mdp
    >>> mdp.test()
    >>> import bimdp
    >>> bimdp.test()

If some test fails, please report it to the `mailing list
<https://lists.sourceforge.net/lists/listinfo/mdp-toolkit-users>`_.  
If you want to help in debugging, start by installing the `py.test
<http://pytest.org/>`_ testing framework.
