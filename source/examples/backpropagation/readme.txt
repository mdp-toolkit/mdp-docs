Backpropagation in a multilayer perceptron based on BiMDP
=========================================================

written by Niko Wilbert

Run the demo_backprop.py file to get a HTML view of the backpropagation.
Since the backpropagation is compatible with batches (the weight-changes are
summed up), so you can use the normal MDP data format. Of course you can
also use non-batch training by only providing single data points (but in
this case the overhead will be quite large).
  