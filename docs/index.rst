..  -*- coding: utf-8 -*-

.. _contents:

Overview of mgcpy_
===================

.. _mgcpy: https://mgc.neurodata.io

mgcpy is a Python package containing tools for multiscale graph correlation and other statistical tests,
that is capable of dealing with high dimensional and multivariate data.

Motivation
----------

Examining and identifying relationships between variables is critical for many scientists to
definitely establishing causality and deciphering these relationships in further studies. To
approach this problem, the most commonly used statistic utilized is Pearson’s Product-Moment
Correlation (Pearson, 1895) but the test fails to address some of the issues that data
scientists face today (Vogelstein et al., 2016). Other tests conventionally used include
"energy statistics" such as Dcorr; kernel-based approaches such as Hilbert Schmidt
Independence Criterion (HSIC) (Gretton and GyĂśrfi, 2010) which has recently been shown to be
equivalent to "energy statistics" (Sejdinovic et al., 2013)(Shen and Vogelstein, 2018);
Heller, Heller, and Gorfine’s test (HHG) (Heller et al., 2012), and many others.
These tests perform empirically well on either high dimensional linear data or low dimensional
nonlinear data.No approach works well on high dimensional nonlinear data, and no approach
addresses issues on how to interpret the data.

Multiscale graph correlation (MGC) attempts to alleviate these issues. The test utilizes
features of other techniques such ask-nearest neighbors, kernel methods, and multiscale
analysis to detect relationships (Vogelstein et al., 2016) in all types of data, including high
dimensional nonlinear data. The test is also computationally efficient, requiring about half or
one third of the number of samples to achieve the same statistical power (Vogelstein et al.,2016).
In addition, the test provides information about the data’s geometry (Vogelsteinet al., 2016),
allowing for more informed decision making of the underlying relationships in the data

About
------

``mgcpy`` aims to be a comprehensive independence testing package including all of the
commonly used independence tests as mentioned above and additional functionality such as
two sample independence testing and a novel random forest-based independence test. These
tests are not only included to benchmark MGC but to have a convenient location for users if
they would prefer to utilize those tests instead. The package utilizes a simple class structure
to enhance usability while also allowing easy extension of the package for developers. The
package can be installed on all major platforms (e.g. BSD, GNU/Linux, OS X, Windows)from
Python Package Index (PyPI) and GitHub.


Free software
-------------

``mgcpy`` is free software; you can redistribute it and/or modify it under the
terms of the :doc:`Apache-2.0 </license>`.  We welcome contributions.
Join us on `GitHub <https://github.com/neurodata/mgcpy>`_.


Documentation
=============

``mgcpy`` is a hypothesis testing package in python.

.. toctree::
   :maxdepth: 1

   install
   reference/index
   tutorial
   license

.. toctree::
   :maxdepth: 1
   :caption: Useful Links

   mgcpy @ GitHub <https://github.com/neurodata/mgcpy/>
   mgcpy @ PyPI <https://pypi.org/project/mgcpy/>
   Issue Tracker <https://github.com/neurodata/mgcpy/issues/>

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
