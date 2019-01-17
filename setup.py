# adopted from https://github.com/neurodata/graspy (GraSPy)

import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install
from subprocess import check_output, call
from Cython.Build import cythonize
from sys import platform
import numpy

PACKAGE_NAME = 'mgcpy'
DESCRIPTION = 'A set of tools in Python for multiscale graph correlation and other statistical tests'
with open('README.md', encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()
AUTHOR = 'Satish Palaniappan, Bear Xiong, Sambit Panda, Sandhya Ramachandran, Ananya Swaminathan, Richard Guo'
AUTHOR_EMAIL = 'spalani2@jhu.edu'
URL = 'https://github.com/neurodata/mgcpy'
MINIMUM_PYTHON_VERSION = 3, 4  # Minimum of Python 3.4

REQUIRED_PACKAGES = ["numpy>=1.14.5", "scipy>=1.1.0", "pytest>=3.8.0", "pycodestyle>=2.3.1", "h5py>=2.7.1",
                     "Cython==0.29", "flake8>=3.5.0", "pytest-cov>=2.6.0", "coveralls>=1.5.1", "Sphinx>=1.8.2"]
VERSION = '0.1.1'


def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))


check_python_version()

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    install_requires=REQUIRED_PACKAGES,
    url=URL,
    license='Apache License 2.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
    ext_modules=cythonize(["mgcpy/independence_tests/mgc/local_correlation.pyx",
                           "mgcpy/independence_tests/mgc/distance_transform.pyx"],
                          compiler_directives={'embedsignature': True}),
    include_dirs=[numpy.get_include()],
    packages=find_packages()
)
