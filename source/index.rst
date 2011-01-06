.. toctree::
   :hidden:
   :maxdepth: 3

   install.rst
   development.rst
   how_to_cite_mdp.rst
   tutorial/tutorial.rst
   examples/examples.rst
   contact.rst

.. admonition:: News

  05.01.2010
      MDP 3.0 released!

      Several new exciting features.

      Get the full list of `changes since last release`__.

__ http://mdp-toolkit.git.sourceforge.net/git/gitweb.cgi?p=mdp-toolkit/mdp-toolkit;a=blob_plain;f=CHANGES;hb=HEAD

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
  <tutorial>`/`pdf <http://downloads.sourceforge.net/mdp-toolkit/MDP-tutorial.pdf?download>`_
* :ref:`Full list <node_list>` of implemented algorithms
* Typical usage :ref:`examples`
* `API Index <api/index.html>`_

.. include:: tutorial/using_mdp_is_as_easy.rst
