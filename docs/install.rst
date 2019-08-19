Install
=======


Below we assume you have the default Python environment already configured on
your computer and you intend to install ``mgcpy`` inside of it.  If you want
to create and work with Python virtual environments, please follow instructions
on `venv <https://docs.python.org/3/library/venv.html>`_ and `virtual
environments <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_.

First, make sure you have the latest version of ``pip`` (the Python package manager)
installed. If you do not, refer to the `Pip documentation
<https://pip.pypa.io/en/stable/installing/>`_ and install ``pip`` first.

Install the released version
----------------------------

Install the current release of ``mgcpy`` with ``pip``::

    $ pip install mgcpy

To upgrade to a newer release use the ``--upgrade`` flag::

    $ pip install --upgrade mgcpy

If you do not have permission to install software systemwide, you can
install into your user directory using the ``--user`` flag::

    $ pip install --user mgcpy

Alternatively, you can manually download ``mgcpy`` from
`GitHub <https://github.com/neurodata/mgcpy/releases>`_  or
`PyPI <https://pypi.python.org/pypi/mgcpy>`_.
To install one of these versions, unpack it and run the following from the
top-level source directory using the Terminal::

    $ pip install .

Install from Github
-------------------

To install from Github, run the following from the top-level source directory
using the Terminal::

    $ git clone https://github.com/neurodata/mgcpy
    $ cd mgcpy
    $ python3 setup.py install


- ``sudo``, if required
- ``python3 setup.py build_ext --inplace  # for cython``, if you want to test in-place, first execute this

Setting up the development environment
--------------------------------------

- To build image and run from scratch:

  - Install [docker](https://docs.docker.com/install/)
  - Build the docker image, ``docker build -t mgcpy:latest .``

    - This takes 10-15 mins to build
  - Launch the container to go into mgcpy's dev env, ``docker run -it --rm --name mgcpy-env mgcpy:latest``
- Pull image from Dockerhub and run:

  - ``docker pull tpsatish95/mgcpy:latest`` or ``docker pull tpsatish95/mgcpy:development``
  - ``docker run -it --rm -p 8888:8888 --name mgcpy-env tpsatish95/mgcpy:latest``


- To run demo notebooks (from within Docker):

  - ``cd demos``
  - ``jupyter notebook --ip 0.0.0.0 --no-browser --allow-root``
  - Then copy the url it generates, it looks something like this: ``http://(0de284ecf0cd or 127.0.0.1):8888/?token=e5a2541812d85e20026b1d04983dc8380055f2d16c28a6ad``
  - Edit this: ``(0de284ecf0cd or 127.0.0.1)`` to: ``127.0.0.1``, in the above link and open it in your browser
  - Then open ``mgc.ipynb``

- To mount/load local files into docker container:

  - Do ``docker run -it --rm -v <local_dir_path>:/root/workspace/ -p 8888:8888 --name mgcpy-env tpsatish95/mgcpy:latest``, replace ``<local_dir_path>`` with your local dir path.
  - Do ``cd ../workspace`` when you are inside the container to view the mounted files. The **mgcpy** package code will be in ``/root/code`` directory.

Python package dependencies
---------------------------
mgcpy requires the following packages:

- numpy
- scikit-learn
- scipy
- Cython
- pandas
- h5py
- seaborn


Hardware requirements
---------------------
`mgcpy` package requires only a standard computer with enough RAM to support the in-memory operations.

OS Requirements
---------------
This package is supported for *macOS* and partly on *Linux*.


Testing
-------
mgcpy uses the Python ``pytest`` testing package.  If you don't already have
that package installed, follow the directions on the `pytest homepage
<https://docs.pytest.org/en/latest/>`_.
