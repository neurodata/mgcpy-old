The R version is available on CRAN and https://github.com/neurodata/mgc-r.
The MATLAB version is available at https://github.com/neurodata/mgc-matlab.

# mgcpy

[![Coverage Status](https://coveralls.io/repos/github/neurodata/mgcpy/badge.svg?branch=master)](https://coveralls.io/github/neurodata/mgcpy?branch=master)
[![Build Status](https://travis-ci.com/neurodata/mgcpy.svg?branch=master)](https://travis-ci.com/neurodata/mgcpy)
[![PyPI](https://img.shields.io/pypi/v/mgcpy.svg)](https://pypi.org/project/mgcpy/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/mgcpy.svg)](https://pypi.org/project/mgcpy/)
[![DockerHub](https://img.shields.io/docker/automated/tpsatish95/mgcpy.svg)](https://hub.docker.com/r/tpsatish95/mgcpy/)
[![DOI](https://zenodo.org/badge/147731955.svg)](https://zenodo.org/badge/latestdoi/147731955)
[![Documentation Status](https://readthedocs.org/projects/mgcpy/badge/?version=latest)](https://mgcpy.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)
[![Code Climate](https://api.codeclimate.com/v1/badges/51ac28d51f1474bf3567/maintainability)](https://codeclimate.com/github/neurodata/mgcpy/maintainability)

`mgcpy` is a Python package containing tools for independence testing using multiscale graph correlation and other statistical tests, that is capable of dealing with high dimensional and multivariate data.

- [Overview](#overview)
- [Documentation](#documentation)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Setting up the development environment](#setting-up-the-development-environment)
- [License](#license)
- [Issues](https://github.com/neurodata/mgcpy/issues)

# Overview
``mgcpy`` aims to be a comprehensive independence testing package including all of the commonly used independence tests as mentioned above and additional functionality such as two sample independence testing and a novel random forest-based independence test. These tests are not only included to benchmark MGC but to have a convenient location for users if they would prefer to utilize those tests instead. The package utilizes a simple class structure to enhance usability while also allowing easy extension of the package for developers. The package can be installed on all major platforms (e.g. BSD, GNU/Linux, OS X, Windows)from Python Package Index (PyPI) and GitHub.

# Documenation
The official documentation with usage is at: https://mgc.neurodata.io/
ReadTheDocs: https://mgcpy.readthedocs.io/en/latest/

# System Requirements
## Hardware requirements
`mgcpy` package requires only a standard computer with enough RAM to support the in-memory operations.

## Software requirements
### OS Requirements
This package is supported for *macOS* and *Linux*. The package has been tested on the following systems:
+ macOS: Mojave (10.14.1)
+ Linux: Ubuntu 16.04

### Python Dependencies
`mgcpy` mainly depends on the Python scientific stack.

```
numpy
scipy
Cython
scikit-learn
pandas
seaborn
```

# Installation Guide:

### Install from PyPi
```
pip3 install mgcpy
```

### Install from Github
```
git clone https://github.com/neurodata/mgcpy
cd mgcpy
python3 setup.py install
```
- `sudo`, if required
- `python3 setup.py build_ext --inplace  # for cython`, if you want to test in-place, first execute this

# Setting up the development environment:
- To build image and run from scratch:
  - Install [docker](https://docs.docker.com/install/)
  - Build the docker image, `docker build -t mgcpy:latest .`
    - This takes 10-15 mins to build
  - Launch the container to go into mgcpy's dev env, `docker run -it --rm --name mgcpy-env mgcpy:latest`
- Pull image from Dockerhub and run:
  - `docker pull tpsatish95/mgcpy:latest` or `docker pull tpsatish95/mgcpy:development`
  - `docker run -it --rm -p 8888:8888 --name mgcpy-env tpsatish95/mgcpy:latest`


- To run demo notebooks (from within Docker):
  - `cd demos`
  - `jupyter notebook --ip 0.0.0.0 --no-browser --allow-root`
  - Then copy the url it generates, it looks something like this: `http://(0de284ecf0cd or 127.0.0.1):8888/?token=e5a2541812d85e20026b1d04983dc8380055f2d16c28a6ad`
  - Edit this: `(0de284ecf0cd or 127.0.0.1)` to: `127.0.0.1`, in the above link and open it in your browser
  - Then open `mgc.ipynb`

- To mount/load local files into docker container:
  - Do `docker run -it --rm -v <local_dir_path>:/root/workspace/ -p 8888:8888 --name mgcpy-env tpsatish95/mgcpy:latest`, replace `<local_dir_path>` with your local dir path.
  - Do `cd ../workspace` when you are inside the container to view the mounted files. The **mgcpy** package code will be in `/root/code` directory.


## MGC Algorithm's Flow
![MGCPY Flow](https://raw.githubusercontent.com/neurodata/mgcpy/master/MGCPY.png)

## Power Curves
- Recreated Figure 2 in https://arxiv.org/abs/1609.05148, with the addition of MDMR and Fast MGC
![Power Curves](https://raw.githubusercontent.com/neurodata/mgcpy/master/power_curves_dimensions.png)

# License

This project is covered under the **Apache 2.0 License**.
