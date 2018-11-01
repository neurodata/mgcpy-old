# mgcpy

[![Coverage Status](https://coveralls.io/repos/github/NeuroDataDesign/mgcpy/badge.svg?branch=master)](https://coveralls.io/github/NeuroDataDesign/mgcpy?branch=master)
[![Build Status](https://travis-ci.com/NeuroDataDesign/mgcpy.svg?branch=master)](https://travis-ci.com/NeuroDataDesign/mgcpy)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<<<<<<< HEAD
"mgcpy" is a Python package containing tools for multiscale graph correlation and other statistical tests, that is capable of dealing with high dimensional and multivariate data. 

It contains statistical tests including dHSIC, DCORR, HHG, RVCorr, MDMR, and MGC.
=======
`mgcpy` is a Python package containing tools for multiscale graph correlation and other statistical tests, that is capable of dealing with high dimensional and multivariate data.
>>>>>>> 63454719ad124bd1896cc779da95cef7ba8923f8

## Installation Guide:

### Install from PyPi
```
pip3 install mgcpy
```

### Install from Github
```
git clone https://github.com/NeuroDataDesign/mgcpy
cd mgcpy
python3 setup.py install
```
- `sudo`, if required
- `python3 setup.py build_ext --inplace  # for cython`, if you want to test in-place, first execute this

## MGC Algorithm's Flow
![MGCPY Flow](MGCPY.png)

## License

This project is covered under the **Apache 2.0 License**.
