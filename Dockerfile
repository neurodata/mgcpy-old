FROM ubuntu:16.04

# set maintainer
LABEL maintainer="spalani2@jhu.edu"

# update
RUN apt-get update && apt-get -y upgrade

# install packages
RUN apt-get install -y \
    cmake \
    cpio \
    gfortran \
    libpng-dev \
    freetype* \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    software-properties-common\
    git \
    man \
    wget

# install python3
RUN add-apt-repository ppa:jonathonf/python-3.6
RUN apt-get update
RUN apt-get install -y \
  python3.6 \
  python3.6-dev
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.6 get-pip.py

RUN ln -s /usr/bin/python3.6 /usr/local/bin/python

# install MKL
# RUN cd /tmp && \
#   wget http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/11306/l_mkl_2017.2.174.tgz && \
#   tar -xzf l_mkl_2017.2.174.tgz && \
#   cd l_mkl_2017.2.174 && \
#   sed -i 's/ACCEPT_EULA=decline/ACCEPT_EULA=accept/g' silent.cfg && \
#   ./install.sh -s silent.cfg && \
#   cd .. && \
#   rm -rf *
# RUN echo "/opt/intel/mkl/lib/intel64" >> /etc/ld.so.conf.d/intel.conf && \
#   ldconfig && \
#   echo ". /opt/intel/bin/compilervars.sh intel64" >> /etc/bash.bashrc

# install numpy with MKL
# RUN pip install Cython

# RUN cd /tmp && \
#  git clone https://github.com/numpy/numpy.git numpy && \
#  cd numpy && \
#  cp site.cfg.example site.cfg && \
#  echo "\n[mkl]" >> site.cfg && \
#  echo "include_dirs = /opt/intel/mkl/include/intel64/" >> site.cfg && \
#  echo "library_dirs = /opt/intel/mkl/lib/intel64/" >> site.cfg && \
#  echo "mkl_libs = mkl_rt" >> site.cfg && \
#  echo "lapack_libs =" >> site.cfg && \
#  python setup.py build --fcompiler=gnu95 && \
#  python setup.py install && \
#  cd .. && \
#  rm -rf *

# install scipy
# RUN cd /tmp && \
#  git clone https://github.com/scipy/scipy.git scipy && \
#  cd scipy && \
#  python setup.py build && \
#  python setup.py install && \
#  cd .. && \
#  rm -rf *

# change working directory
RUN mkdir /root/workspace/
WORKDIR /root/workspace/

# install python requirements
COPY requirements.txt /root/workspace/requirements.txt
RUN pip install -r requirements.txt
RUN pip install jupyter
RUN pip install matplotlib
RUN pip install pycodestyle

# copy the mgcpy code into the container
COPY . /root/workspace/

# setup pep8 guidelines (restricts push when pep8 is violated)
RUN rm ./.git/hooks/pre-commit
RUN chmod 777 install-hooks.sh
RUN ./install-hooks.sh

# install mgcpy
RUN python setup.py build_ext --inplace

# add mgcpy to PYTHONPATH for dev purposes
RUN echo "export PYTHONPATH='${PYTHONPATH}:/root/workspace'" >> ~/.bashrc

# test if mgcpy is correctly installed
RUN py3clean .
RUN pytest

# launch terminal
CMD ["/bin/bash"]
