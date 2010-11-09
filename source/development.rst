.. _maintainers:

********************************
Project development & guidelines
********************************

-----------
Maintainers
-----------

MDP has been originally written by `Pietro Berkes`_ and `Tiziano Zito`_
at the `Institute for Theoretical Biology <http://itb.biologie.hu-berlin.de/>`_
of the `Humboldt University <http://www.hu-berlin.de/>`_, Berlin in 2003.

Current maintainers are:

*   `Pietro Berkes`_
*   `Rike-Benjamin Schuppner`_
*   `Niko Wilbert`_
*   `Tiziano Zito`_
*   `Zbigniew Jędrzejewski-Szmek`_

`Yaroslav Halchenko`_ maintains the python-mdp_ Debian package,
`Maximilian Nickel`_ maintains the py25-mdp-toolkit_ MacPorts package.

.. _`Pietro Berkes`: http://people.brandeis.edu/~berkes
.. _`Niko Wilbert`: http://itb.biologie.hu-berlin.de/~wilbert
.. _`Tiziano Zito`: http://www.cognition.tu-berlin.de/menue/members/tiziano_zito
.. _`Rike-Benjamin Schuppner`: http://www.bccn-berlin.de/People/home/?contentId=686
.. _`Zbigniew Jędrzejewski-Szmek`: http://dimer.fuw.edu.pl/Members/ZbyszekJSzmek
.. _`Yaroslav Halchenko`: http://www.onerussian.com
.. _python-mdp: http://packages.debian.org/python-mdp
.. _`Maximilian Nickel`: http://2manyvariables.inmachina.com
.. _py25-mdp-toolkit: http://trac.macports.org/browser/trunk/dports/python/py25-mdp-toolkit/Portfile

For comments, patches, feature requests, support requests, and bug reports
(if any) you can use the users’ `mailing list`_.

.. _`mailing list`: https://lists.sourceforge.net/mailman/listinfo/mdp-toolkit-users

If you want to contribute some code or a new algorithm, please do not
hesitate to submit it!

------------
Contributors
------------
In this final section we want to thank all users who have contributed
code and bug reports to the MDP project. Strictly in alphabetical order:

- `Gabriel Beckers <http://www.gbeckers.nl/>`_
- Sven Dähne
- Alberto Escalante
- `Farzad Farkhooi <http://www.bccn-berlin.de/People/farkhooi>`_
- Mathias Franzius
- `Michael Hanke <http://apsy.gse.uni-magdeburg.de/main/index.psp?page=hanke/main&lang=en&sec=0>`_
- `Konrad Hinsen <http://dirac.cnrs-orleans.fr/~hinsen/>`_
- Christian Hinze
- `Samuel John <http://www.samueljohn.de/>`_
- Susanne Lezius
- `Michael Schmuker <http://userpage.fu-berlin.de/~schmuker/>`_
- `Benjamin Schrauwen <http://snn.elis.ugent.be/benjamin>`_
- `Henning Sprekeler <http://lcn.epfl.ch/~sprekele>`_
- `Jake VanderPlas <http://www.astro.washington.edu/vanderplas/>`_

.. numpy
.. _numpy: http://numpy.scipy.org
.. _`numpy github`: http://github.com/numpy/numpy
.. _`numpy mailing list`: http://mail.scipy.org/mailman/listinfo/numpy-discussion

.. scipy
.. _scipy: http://www.scipy.org
.. _`scipy github`: http://github.com/scipy/scipy
.. _`scipy mailing list`: http://mail.scipy.org/mailman/listinfo/scipy-dev

.. python
.. _python: http://www.python.org

=================================================================

----------------------
Development guidelines
----------------------

-----------
Source code
-----------

MDP is currently maintained by a core team of 5 developers, but it is
open to user contributions. Users have already contributed some of the
nodes, and more contributions are currently being reviewed for
inclusion in future releases of the package. The package development
can be followed online on the public git code `repositories`_ or
cloned with::

    git clone git://mdp-toolkit.git.sourceforge.net/gitroot/mdp-toolkit/mdp-toolkit
    git clone git://mdp-toolkit.git.sourceforge.net/gitroot/mdp-toolkit/docs
    git clone git://mdp-toolkit.git.sourceforge.net/gitroot/mdp-toolkit/examples
    git clone git://mdp-toolkit.git.sourceforge.net/gitroot/mdp-toolkit/contrib

.. _repositories: http://mdp-toolkit.git.sourceforge.net

Questions, bug reports, and feature requests are typically handled by
the user `mailing list`_.

Supported dependency versions
-----------------------------

- `Python`_: 2.5, 2.6, 2.7, 3.1, 3.2
- `Numpy`_: 1.4, 1.5

- `Scipy`_: 0.6 and 0.7

Core MDP must depend only on Python and Numpy.


Information for new developers
------------------------------

Dear new MDP developer, we’ll try here to summarize some policies
and best-practices specific to new developers. You should also follow
the general style guidelines specified below, which are applicable to
all developers.

- Create an account on sourceforge.net and tell us your username
  there, so that we can add you to the list of developers and give
  you access to our git repositories

- Since our migration to git, the repository setup consists of
  four separate repositories:

  * mdp-toolkit
  * docs
  * examples
  * contrib

- Please only commit code in contrib repository.
  If your code contributions should need modification somewhere else
  in the MDP code base, write to
  mdp-toolkit-devel@lists.sourceforge.net
  for assistance and instructions how to proceed.
  For simple fixes that don’t need much discussion, you can also send
  a mail patch to the list using ``git format-patch`` or similar.

- The only exception to the previous rule are the tests for your code. They
  should be added in
  ``mdp-toolkit/mdp/test/test_contrib.py``
  or in some yet-to-be-defined place in the contrib repository.
  Look how other contrib nodes are tested in the ``ContribTestSuite``
  class in this file, and make sure your tests fit within that
  framework. Be particularly aware of the automatic testing of
  setting and consistency of ``input_dim``, ``output_dim`` and ``dtype``.

- Your code contribution should not have any additional
  dependencies, i.e. they should require only the numpy module to be
  installed. If your code requires some other module, e.g. scipy or
  C++ compilation, ask
  mdp-toolkit-devel@lists.sourceforge.net
  for assistance.

Also see the `General style guidelines`_ and
`Development on Microsoft Windows`_ below.

======================================================================

Development process
-------------------

Development takes place on the ``master`` branch, but it doesn't mean
that everything should be immediately commited there.

Small commits and bugfixes and the like should go immediately on the
main branch, if the commiter thinks that nothing will be broken by the
patch::

    git checkout master
    # make a small fix :)
    sed -ir s/develepement/development/g development_process.rst
    git add development_process.rst
    git commit -m 'FIX: correct spelling of development'

More complicated commits should go on a feature branch::

    git checkout -b my_new_feature
    <do some changes>
    git add <some-file> <some-other-file>
    git commit -m 'NEW: add subfeature-1'
    <do some more changes>
    git commit -m 'NEW: implement this and that'

When a developer wants to show the branch to other people, she should
push it into the main repo::

    git push origin my_new_feature


Temporary branches
``````````````````

If you are about to test something and you’ve got the idea that your
code won’t last long in the repository, (maybe you want to show your
code to another developer or you want to just check, if you can commit
to the server,) you should create another branch for that, the same as
for any new feature.

The advantage is, that it keeps our master branch clean from all those
‘testing some really strange new stuff – please have a look’ commits,
which are likely to be reverted again. When you feel good about your
commit, you can cherry-pick or merge the good stuff to master.


Merging feature branches back into the `master` branch
``````````````````````````````````````````````````````

Development is consensus based, so new features should be posted for
review and gain acceptance before being merged back into the main
branch. After the decision to merge has been made:

#. Check that all tests pass on the feature branch. Ideally, the branch
   should already include tests for all code it introduces or
   significantly changes.

   Some things to test in special circumstances:

   - If the code does anything version specific, it should be tested on
     all supported python versions (c.f. `Supported Dependency Versions`_)::

         python2.5 /usr/bin/py.test
         python2.6 /usr/bin/py.test
         python2.7 /usr/bin/py.test
         python3.1 setup.py build
         (cd build/py3k && py.test-3.1)
         (cd build/py3k && python3.2 /usr/bin/py.test-3.1)

     TODO: add windows and mac equivalents

   - If the code does anything platform specific if should also be
     tested on Windows.

   Before merging also make sure that the master branch passes tests :)

#. The merge should be performed in a way that preserves the history
   of the branch::

       git checkout master
       git merge --no-ff my_new_feature

   The merge commit should retain the name of the branch in the
   message. E.g. a commit with a message *Merge branch my_new_feature*
   is OK, commit with a message
   *Merge commit 1234567890123456789012345678901234567890* is not so good.

#. After merging, tests should also pass.

   If tests fail and the failures are caused by a problem with the
   merge, the merge commit should be amended::

       <fix code>
       py.test ...
       git commit --amend -a

   If the changes introduced in the branch simply uncovered problems in
   other parts of the codebase, the fixes can be committed as separate
   changesets.

#. Only when tests after the merge execute satisfactorily, changes
   should be pushed to sourceforge. The old branch can be deleted.::

       git push origin master :my_new_feature

Git commit messages
-------------------

Commit messages are supposed to start with a prefix that specifies the
type of change.

* DOC: — documentation
* FIX: — fixes something
* ERF: — enhancement, refactoring
* NEW: — a new feature
* OTH: — other

The message should consist of a short summary (up to about 70
characters) and a longer explanation after an empty line. The summary
messages will are used to generate a changelog for distribution
tarballs.

Notes in source code
--------------------

Parts of code requiring special attention can be marked with

* ``FIXME``
* ``TODO``
* ``XXX`` or ``???`` (a question)
* ``YYY`` (answer to ``XXX``)
* ``NOTE`` (a random comment)
* ``WARNING`` (a warning for developers)


History rewriting
-----------------

The developer that created a feature branch is free to rewrite the
history of the branch if she finds it reasonable. SF is currently
configured to deny non-fast-forward pushes, but this can be
cimcurvented by first deleting the branch, and then pushing a new
version::

    # do some history cleaning
    git rebase -i $(git merge-base origin/master my_new_feature)
    # nuke the branch on sf
    git push origin :my_new_feature
    # upload a new version of the branch
    git push origin my_new_feature

If multiple developers wants to cooperate on ``feature_branch``, they
should agree between themselves on a history rewriting policy.

------------------------
General style guidelines
------------------------

- Remember to set the supported dtypes for your nodes.
  Example of a node supporting only single and double precision:
  * ``SFANode`` in mdp-toolkit/mdp/nodes/sfa_nodes.py
  Example of a node supporting almost every dtype:
  * ``HitParadeNode`` in mdp-toolkit/mdp/nodes/misc_nodes.py

- If setting ``input_dim``, ``output_dim`` or ``dtype`` has side
  effects, remember to implement that in the ``_set_input_dim``,
  ``_set_output_dim``, ``_set_dtype`` functions.  Several examples are
  available in ``mdp-toolkit/mdp/nodes/``

- Your code should strictly follow the PEP 8 coding convenctions
  (http://www.python.org/dev/peps/pep-0008/). Note that some older code
  sections in MDP do not follow PEP 8 100%, but when the opportunity arrises
  (e.g., when we make changes in the code) we are improving this. So new code
  should always follow PEP 8. Additional style guidelines can be learned from
  the famous 'Code like a Pythonista' presentation at
  http://python.net/~goodger/projects/pycon/2007/idiomatic/handout.html.

- Always import numpy in your code as::

    from mdp import numx

  ``numx`` is a placeholder we use to automatically import scipy
  instead of numpy when scipy is installed on the system.

- Only raise NodeException. If you need custom exceptions, derive
  them from ``mdp.NodeException``.

- Your nodes needs to pass the automatic tests for setting and
  consistency of ``input_dim``, ``output_dim`` and ``dtype`` *and* at
  least one functional test, which should test the algorithm possibly
  in a non-trivial way and compare its results with exact data you can
  derive analytically. If the latter is not possible, you should
  compare results and expected data within a certain precision. Look
  for example at ``testPCANode`` in ``mdp-toolkit/mdp/test/test_PCANode.py``.

- You nodes must have telling and explicit doc-strings. In
  particular, the class doc-string must cite references (if any) for
  the algorithm, and list the internal attributes of interest for
  the user. Any method not belonging to the base ``Node`` class must be
  clearly documented in its doc-string. Error messages must give an
  hint to the user what’s wrong and possible ways around the
  problem. Any non trivial algorithmic step in the code must be
  commented, so that other developers understand what’s going on. If
  you have doubts, mark the code with::

    #???

  If you think a better implementation is possible or additional
  work is needed, mark the code with::

    #TODO

  Have a look at the ``SFANode`` implementation for an example.

- When you commit your code *always* provide a meaningful log
  message: it will be mailed automatically to all other developers!

- This list is far from being complete, please let us know your
  comments and remarks :-)

========================================================================

--------------------------------
Development on Microsoft Windows
--------------------------------

If you want to develop on a Windows system you might run into some issues
with git. Here is what we use for git on Windows:

* Install the msysgit git client.
* If you don't like working on the command line there are several graphical
  user interfaces available, the commercial SmartGit currently seems
  to work best (there is a free version for non-commercial use).

If you want to use the Eclipse IDE (with PyDev) here is what you can do:

* You can install the EGit plugin for Eclipse, but this is not yet stable. So
  you might want to use the command line or SmartGit for most actions.
* Create a new PyDev project for each MDP git repo you want to work on. Clone
  the git repository to some arbitrary location and then move all the content
  (including the hidden .git folder) to the root of the corresponding project
  (EGit currently will not work if the .git is in some subdirectory).
* Right-click on the project and select Team -> share to connect the git
  information to EGit.
