
--------


Installation
------------

**Requirements:**
`Python`_ ? 2.4, and `NumPy`_ ? 1.1 or `Scipy`_ ? 0.5.2. If you have `Scipy`_
? 0.7 the `symeig`_ package is not needed anymore for additional speed.

**Download:**
You can download the last MDP release ` here`_.
If you want to live on the bleeding edge, check out the MDP git repositories.
You can either `browse the repositories`_ or clone the ``mdp-toolkit``
repository with: ::
    git clone git://mdp-toolkit.git.sourceforge.net/gitroot/mdp-toolkit
    /mdp-toolkit

and then install as explained below.
Thanks to Yaroslav Halchenko, users of Debian, Ubuntu and derivatives can
install the `python-mdp`_ package.
Thanks to Maximilian Nickel, Mac OS X users using `MacPorts`_ can install the
packages `py25-mdp-toolkit`_ and `py26-mdp-toolkit`_, respectively.
Gentoo users can install the ebuild `sci-mathematics/mdp`_ from the
``science`` overlay.

**Installation:**
Unpack the archive file and change to the project directory or change to the
cloned git repository, and type: ::
    python setup.py install
     If you want to use MDP without installing it on the system Python
     path: ::
    python setup.py install --prefix=/some_dir_in_PYTHONPATH/
     On Debian you can just type: ::
    aptitude update
    aptitude install python-mdp
     On Mac OS X if you use the `MacPorts`_ system, just type: ::
    sudo port install py25-mdp-toolkit
     or: ::
    sudo port install py26-mdp-toolkit
     depending on your favoured version of Python.
On Gentoo you can use your favourite package manager or, alternatively: ::
    emerge layman
    layman -L
    layman -a science
    emerge sci-mathematics/mdp
     On Windows, the installation of the binary distribution is as easy
     as executing the installer and following the instructions.

**Testing:**
If you have successfully installed MDP, you can test your installation in a
Python shell as follows: ::
    >>> import mdp
    >>> mdp.test()
    >>> import bimdp
    >>> bimdp.test()


**Demos:**
All the code examples shown in the `MDP tutorial`_ together with several
other demos can be found in the package installation path in the subdirectory
``demo``.

--------


Maintainers
-----------

MDP has been originally written by `Pietro Berkes`_ and `Tiziano Zito`_ at
the `Institute for Theoretical Biology`_ of the `Humboldt University`_,
Berlin in 2003.

Current maintainers are:

-   `Pietro Berkes`_
-   Rike-Benjamin Schuppner
-   `Niko Wilbert`_
-   `Tiziano Zito`_

`Yaroslav Halchenko`_ maintains the `python-mdp`_ Debian package, `Maximilian
Nickel`_ maintains the ` ``py25-mdp-toolkit```_ MacPorts package.

For comments, patches, feature requests, support requests, and bug reports
(if any) you can use the users `mailing list`_.

If you want to contribute some code or a new algorithm, please do not
hesitate to submit it!

--------


How to cite MDP
---------------

If you use MDP for scientific purposes, you may want to cite it. This is the
official way to do it:

Zito, T., Wilbert, N., Wiskott, L., Berkes, P. (2009)
**Modular toolkit for Data Processing (MDP): a Python data processing frame
work**
Front. Neuroinform. (2008) **2**:8. Homepage: `http://mdp-
toolkit.sourceforge.net`_

You can get the paper `here`_.

If your paper gets published, plase send us a reference (and even a copy if
you don't mind).

.. _MDP Sprint 2010: http://sourceforge.net/apps/mediawiki/mdp-
    toolkit/index.php?title=MDP_Sprint_2010
.. _changes     since last release: CHANGES
.. _git: http://mdp-toolkit.git.sourceforge.net/
.. _presented: EuroScipy2009Talk.pdf
.. _EuroScipy:
    http://www.euroscipy.org/presentations/abstracts/abstract_zito.html
.. _presented: CNS2009Talk.pdf
.. _Python in Neuroscience: http://www.cnsorg.org/2009/workshops.shtml
.. _CNS 2009: http://www.cnsorg.org/2009/
.. _Introduction: tutorial.html#introduction
.. _Full list: tutorial.html#node-list
.. _Tutorial: tutorial.html
.. _pdf: http://prdownloads.sourceforge.net/mdp-
    toolkit/MDP2_6_tutorial.pdf?download
.. _API: docs/api/index.html
.. _Python: http://www.python.org/
.. _NumPy: http://numpy.scipy.org/
.. _Scipy: http://www.scipy.org/
.. _symeig: symeig.html
.. _ here: http://sourceforge.net/projects/mdp-toolkit/files
.. _python-mdp: http://packages.debian.org/python-mdp
.. _MacPorts: http://www.macports.org/
.. _py25-mdp-toolkit:
    http://trac.macports.org/browser/trunk/dports/python/py25-mdp-
    toolkit/Portfile
.. _py26-mdp-toolkit:
    http://trac.macports.org/browser/trunk/dports/python/py26-mdp-
    toolkit/Portfile
.. _sci-mathematics/mdp:
    http://git.overlays.gentoo.org/gitweb/?p=proj/sci.git;a=tree;f=sci-
    mathematics/mdp
.. _Pietro Berkes: http://people.brandeis.edu/~berkes
.. _Tiziano Zito: http://itb.biologie.hu-berlin.de/~zito
.. _Institute for Theoretical Biology: http://itb.biologie.hu-berlin.de/
.. _Humboldt University: http://www.hu-berlin.de/
.. _Niko Wilbert: http://itb.biologie.hu-berlin.de/~wilbert
.. _Yaroslav Halchenko: http://www.onerussian.com/
.. _Maximilian Nickel: http://2manyvariables.inmachina.com
.. _mailing list: https://lists.sourceforge.net/mailman/listinfo/mdp-
    toolkit-users
.. _http://mdp-toolkit.sourceforge.net: http://mdp-
    toolkit.sourceforge.net
.. _here: http://dx.doi.org/10.3389/neuro.11.008.2008
