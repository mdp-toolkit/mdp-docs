.. toctree::
   :hidden:
   :maxdepth: 3

   install.rst
   development.rst
   how_to_cite_mdp.rst
   tutorial/tutorial.rst
   examples/index.rst
   contact.rst   

.. admonition:: News
  
  19.07.2010
    Come and join us at the `MDP sprint <http://sourceforge.net/apps/mediawiki/mdp-toolkit/index.php?title=MDP_Sprint_2010>`_!
    
  14.05.2010
    MDP 2.6 released!
    
    * Several new classifier nodes have been added.
    * A new node extension mechanism makes it possible to dynamically
      add methods or attributes for specific features to node classes,
      enabling aspect-oriented programming in MDP. Several MDP features (like
      parallelization) are now based on this mechanism, and users can add their
      own custom node extensions.
    * BiMDP is a large new package in MDP that introduces bidirectional
      data flows to MDP, including backpropagation and even loops. BiMDP also
      enables the transportation of additional data in flows via messages.
    * BiMDP includes a new flow inspection tool, that runs as as a
      graphical debugger in the webrowser to step through complex flows. It can
      be extended by users for the analysis and visualization of intermediate
      data.
    * As usual, tons of bug fixes

    Get the full list of (`changes since last release <CHANGES>`_).
  
  07.05.2010
    Finalized the migration to `git <http://mdp-toolkit.git.sourceforge.net/git/gitweb-index.cgi>`_.


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
* Typical usage :ref:`examples`
* `API Index <api/index.html>`_

.. include:: tutorial/using_mdp_is_as_easy.rst
