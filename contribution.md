
Contributing to PyCNN
============================

**Note: This document is a 'getting started' summary for contributing code,
documentation, testing, and filing issues. ** Please read this carefully to help make
the code review process go as smoothly as possible and maximize the
likelihood of your contribution being merged.**

How to contribute
-----------------

The preferred workflow for contributing to scikit-learn is to fork the 
[main repository](https://github.com/ankitaggarwal011/PyCNN.git) on
GitHub, clone, and develop on a branch. Steps:

1. Fork the [project repository](https://github.com/ankitaggarwal011/PyCNN.git)
   by clicking on the 'Fork' button near the top right of the page. This creates
   a copy of the code under your GitHub user account.

2. Clone your fork of the scikit-learn repo from your GitHub account to your local disk:

   ```bash
   $ git clone git@github.com:YourLogin/PyCNN.git
   $ cd scikit-learn
   ```
   
3. Create a ``changes`` branch to hold your development changes:

   ```bash
   $ git checkout -b my-changes
   ```
   
   Always use a ``cahnges`` branch. It's good practice to never work on the ``master`` branch!

4. Develop the feature on your changes branch. Add changed files using ``git add`` and then ``git commit`` files:

   ```bash
   $ git add modified_files
   $ git commit
   ```

   to record your changes in Git, then push the changes to your GitHub account with:

   ```bash
   $ git push -u origin my-changes
   ```

5. Go to the GitHub web page of your fork of the PyCNN repo.
Click the 'Pull request' button to send your changes to the project's maintainers for
review. This will send an email to the committers.

(If any of the above seems like magic to you, please look up the 
[Git documentation](https://git-scm.com/documentation) on the web, or ask a friend or another contributor for help.)


New contributor tips
--------------------

A great way to start contributing to PyCNN is to pick an item
from the list of [Easy issues](https://github.com/ankitaggarwal011/PyCNN.git/issues?labels=Easy)
in the issue tracker. Resolving these issues allow you to start
contributing to the project without much prior knowledge. Your
assistance in this area will be greatly appreciated by the more
experienced developers as it helps free up their time to concentrate on
other issues.

