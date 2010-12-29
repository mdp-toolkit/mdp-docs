.. _examples:

Examples
========

.. toctree::
   :hidden:

   logmap/logmap.rst
   lle/lle.rst
   gng/gng.rst
   logo/logo_animation.rst
   bayes_guesser/bayes_guesser.rst
   word_generator/word_generator.rst
   slideshow/slideshow.rst

Here are examples on how to use MDP for typical machine learning
applications:

* :ref:`logmap` — Using Slow Feature Analysis (SFA) for
  processing a non-stationary time series, derived by a logistic map.
* :ref:`gng` — Capture the topological structure of a
  data distribution.
* :ref:`lle` — Approximate data with a low-dimensional surface
  and reduce its dimensionality by learning a mapping to the surface.

* `hinet_html.py`__ — Get the HTML representation for a simple hinet network.
* `benchmark_parallel.py`__ — Simple benchmark to compare the different
  schedulers in MDP.
* :ref:`slideshow` and :ref:`slideshow_double.py` — Created slideshows of
  matplotlib plots, demonstrates the slideshow module in MDP.
* `hinetplaner`__ — Interactive HTML/JS/AJAX based GUI for constructing special
  hinet networks. This is a complicated example which won't teach you much
  about MDP.

__ http://mdp-toolkit.git.sourceforge.net/git/gitweb.cgi?p=mdp-toolkit/examples;a=blob_plain;f=hinet_html.py
__ http://mdp-toolkit.git.sourceforge.net/git/gitweb.cgi?p=mdp-toolkit/examples;a=blob_plain;f=benchmark_parallel.py
__ http://mdp-toolkit.git.sourceforge.net/git/gitweb.cgi?p=mdp-toolkit/examples;a=tree;f=hinetplaner

The following examples use and illustrate BiMDP.

* `bimdp_inverse.py`__ — A simple example on the alternative
  mechanism to inverse a BiFlow.
* `bimdp_hinet_inspection.py`__ — Demonstrates the inspection of
  a BiFlow.
* `bimdp_custom_inspection.py`__ — Customization with maptlotlib
  plots of the BiFlow inspection.
* `bimdp_simple_coroutine.py`__ — Minimal example for coroutine decorator.
* `gradnewton`__ — Use Newton's method for gradient descent
  with the gradient extension.
* `backpropagation`__ — Implement backpropagation for a multi layer
  perceptron.
* `binetdbn`__ — Proof of concept for a Deep Belief Network.

__ http://mdp-toolkit.git.sourceforge.net/git/gitweb.cgi?p=mdp-toolkit/examples;a=blob_plain;f=bimdp_examples/bimdp_inverse.py
__ http://mdp-toolkit.git.sourceforge.net/git/gitweb.cgi?p=mdp-toolkit/examples;a=blob_plain;f=bimdp_examples/bimdp_hinet_inspection.py
__ http://mdp-toolkit.git.sourceforge.net/git/gitweb.cgi?p=mdp-toolkit/examples;a=blob_plain;f=bimdp_examples/bimdp_custom_inspection.py
__ http://mdp-toolkit.git.sourceforge.net/git/gitweb.cgi?p=mdp-toolkit/examples;a=blob_plain;f=bimdp_examples/bimdp_simple_coroutine.py
__ http://mdp-toolkit.git.sourceforge.net/git/gitweb.cgi?p=mdp-toolkit/examples;a=tree;f=gradnewton
__ http://mdp-toolkit.git.sourceforge.net/git/gitweb.cgi?p=mdp-toolkit/examples;a=tree;f=backpropagation
__ http://mdp-toolkit.git.sourceforge.net/git/gitweb.cgi?p=mdp-toolkit/examples;a=tree;f=binetdbn
