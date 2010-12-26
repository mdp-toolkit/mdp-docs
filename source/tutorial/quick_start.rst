Quick Start
===========

.. include:: ../main.rst

.. include:: using_mdp_is_as_easy.rst

MDP requires the numerical Python extensions `NumPy`_ or `SciPy`_. At
import time MDP will select ``scipy`` if available, otherwise
``numpy`` will be loaded. You can force the use of a numerical
extension by setting the environment variable ``MDPNUMX=numpy`` or
``MDPNUMX=scipy``. 

.. admonition:: An important remark
    
   Input array data is typically assumed to be two-dimensional and
   ordered such that observations of the same variable are stored on
   rows and different variables are stored on columns.
