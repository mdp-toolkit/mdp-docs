.. _examples:

Examples
========

.. toctree::
   :hidden:
   
   logmap_src/logmap.rst
   lle_src/lle.rst
   gng_src/gng.rst

Here are examples on how to use MDP for typical machine learning
applications:
   
* :ref:`examples_logmap` — Using Slow Feature Analysis (SFA) for
  processing a non-stationary time series, derived by a logistic map.
* :ref:`examples_gng` — Capture the topological structure of a
  data distribution.
* :ref:`examples_lle` — Approximate data with a low-dimensional surface
  and reduces its dimensionality by learning a mapping to the surface.
  
The following examples are available in mdp examples repository or can
be downloaded seperatly from the last release **ADD LINKS!!!**. Some of
them are explained with docstrings and come with readme files.

* hinet_html.py — Get the HTML representation for a simple hinet network.
* benchmark_parallel.py — Simple benchmark to compare the different
  schedulers in MDP.
* bayes_guesser.py
* word_generator.py
* slideshow.py and slideshow_double.py — Created slideshows of
  matplotlib plots, demonstrates the slideshow module in MDP.
* hinetplaner — Interactive HTML/JS/AJAX based GUI for constructing special
  hinet networks. This is a complicated example which won't teach you much
  about MDP.

The following examples use and illustrate BiMDP.

* bimdp_examples/bimdp_inverse.py — A simple example on the alternative
  mechanism to inverse a BiFlow.
* bimdp_examples/bimdp_hinet_inspection.py — Demonstrates the inspection of
  a BiFlow.
* bimdp_examples/bimdp_custom_inspection.py — Customization with maptlotlib
  plots of the BiFlow inspection.
* bimdp_simple_coroutine.py - Minimal example for coroutine decorator.
* gradnewton — Use Newton's method for gradient descent
  with the gradient extension.
* backpropagation — Implement backpropagation for a multi layer
  perceptron.
* binetdbn — Proof of concept for a Deep Belief Network.

