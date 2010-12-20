.. install:

************
Installation
************

Requirements
============

.. |gE| unicode:: U+2267

* `Python <http://www.python.org/>`_ 2.5/2.6/2.7/3.1/3.2
* `NumPy <http://numpy.scipy.org/>`_ |gE| 1.1 or `Scipy <http://www.scipy.org/>`_ |gE| 0.5.2.

License
=======
MDP is distributed under the open source `BSD license <http://www.opensource.org/licenses/bsd-license.php>`_. 

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


