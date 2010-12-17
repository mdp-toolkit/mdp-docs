.. toctree::
   :hidden:

   news.rst
   install.rst
   development.rst
   how_to_cite_mdp.rst
   tutorial_src/tutorial.rst
   examples_src/examples.rst

**Modular toolkit for Data Processing (MDP)** is a Python data processing
framework.

From the user's perspective, MDP is a collection of supervised and
unsupervised learning algorithms and other data processing units that can be
combined into data processing sequences and more complex feed-forward network
architectures.

From the scientific developer's perspective, MDP is a modular framework,
which can easily be expanded. The implementation of new algorithms is easy
and intuitive. The new implemented units are then automatically integrated
with the rest of the library.

The base of available algorithms is steadily increasing and includes, to name
but the most common, Principal Component Analysis (PCA and NIPALS), several
Independent Component Analysis algorithms (CuBICA, FastICA, TDSEP, JADE, and
XSFA), Slow Feature Analysis, Gaussian Classifiers, Restricted Boltzmann
Machine, and Locally Linear Embedding.

To learn more about MDP:

* Tutorial: :ref:`html
  <tutorial>`/`pdf <http://prdownloads.sourceforge.net/mdp-toolkit/MDP2_6_tutorial.pdf?download>`_
* :ref:`Full list <node-list>` of implemented algorithms
* Typical usage :ref:`examples <examples>`
* `API Index <api/index.html>`_

Using MDP is as easy as: ::

    >>> import mdp
    >>> # perform pca on some data x
    ...
    >>> y = mdp.pca(x)
    >>> # perform ica on some data x using single precision
    ...
    >>> y = mdp.fastica(x, dtype='float32')


