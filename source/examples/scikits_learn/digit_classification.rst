.. _digit_classification:

Handwritten digits classification with MDP and scikits.learn
============================================================

If you have the library `scikits.learn <http://scikit-learn.sourceforge.net>`_
installed on your machine, MDP will automatically wrap the algorithms defined
there, and create a new series of Nodes, whose name ends with
`ScikitsLearnNode`.

In this example, we show how it is possible to merge MDP nodes and flows
and a scikits.learn classifier to perform handwritten digit classification.

One of the nice features of scikits.learn is that it provides access to
several classical dataset. First of all, we will load the `digits` dataset

    >>> import mdp
    >>> import numpy

    >>> import scikits.learn as sl
    >>> from scikits.learn import datasets

    >>> digits = datasets.load_digits()
    >>> images = digits.images.astype('f')
    >>> labels = digits.target
    >>> data = digits.images.reshape((images.shape[0],
    ...                               numpy.prod(images.shape[1:])))

and divide it in a training and a test set

    >>> # number of digits to be used as training data (2/3 of total)
    >>> ntrain = images.shape[0] // 3 * 2
    >>> train_data = [data[:ntrain, :]]
    >>> train_data_with_labels = [(data[:ntrain, :], labels[:ntrain])]
    >>> test_data = data[ntrain:, :]
    >>> test_labels = labels[ntrain:]

[here show examples of the digits]
 
For our handwritten digits classification application, we build a flow that
performs these steps:
 
1) reduce the dimensionality of the data to 25

2) expand the data in the space of polynomials of degree 3, i.e., augment the
   pixel data (x_1, x_2, ...) with monomial of order 2 and three, like
   x_1 * x_2^2 or x_1 * x_3 . This is a common trick to transform a linear
   algorithm (in this case, the one in step 4) in a non-linear algorithm
   
3) reduce the dimensionality of the data again, keeping 99% of the
   variance in the expanded data
   
4) perform Fisher Discriminant Analysis (FDA), a supervised algorithm that
   finds a projection of the data that maximize the variance between labels,
   and minimize the variance within labels. In other words, the algorithm
   tries to form well separated clusters for each label.

5) classify the digit using the Support Vector Classification algorithm
   defined in scikits.learn

In the application, the data type is set to single precision to spare memory

    >>> flow = mdp.Flow([mdp.nodes.PCANode(output_dim=25, dtype='f'),
    ...                  mdp.nodes.PolynomialExpansionNode(3),
    ...                  mdp.nodes.PCANode(output_dim=0.99),
    ...                  mdp.nodes.FDANode(output_dim=9),
    ...                  mdp.nodes.SVCScikitsLearnNode(kernel='rbf')], verbose=True)
 
Note how it is possible to set parameters in the the scikits.learn algorithms
simply by using the corresponding keyworg argument. In this case, we use
Radial Basis Function kernels for the classifier.

We're ready to train our algorithms on the training data set:

    >>> flow.train([train_data, None, train_data,
    ...             train_data_with_labels, train_data_with_labels])
    >>> # print the final state of the nodes
    >>> print repr(flow)

Finally, we can execute the application on the test data set, and compute
the error rate:

   >>> # set the execution behavior of the last node to return labels
   >>> flow[-1].execute = flow[-1].label
   >>> 
   >>> # get test labels
   >>> prediction = flow(test_data)
   >>> # percent error
   >>> error = ((prediction.flatten() != test_labels).astype('f').sum()
   >>>          / (images.shape[0] - ntrain) * 100.)
   >>> print 'percent error:', error
   [rispost]
