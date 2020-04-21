FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ENV PATH /usr/lib/ccache:$PATH

RUN apt-get -y update && apt-get -y upgrade && \
  apt-get install -y ccache cmake ca-certificates libjpeg-dev libpng-dev curl g++ gfortran git libhdf5-dev libopencv-dev && \
  apt-get clean

RUN apt-get install -y language-pack-ja-base language-pack-ja && \
  echo 'export LANG=ja_JP.UTF-8' >> ~/.bashrc && \
  echo 'export LC_CTYPE=ja_JP.UTF-8' >> ~/.bashrc

RUN apt-get -y -q install python-tk python-pip&& \
    apt-get -y install vim && \
    apt-get -y install wget

RUN pip install --upgrade pip

RUN pip install -U setuptools
RUN pip install -U cython
RUN pip install -U h5py
RUN pip install -U scipy
RUN pip install -U matplotlib
RUN pip install -U hacking
RUN pip install -U nose
RUN pip install -U mock
RUN pip install -U coverage
RUN pip install -U scikit-learn
RUN pip install -U pillow
RUN pip install -U opencv-python
RUN pip install -U torch
RUN pip install -U torchvision
RUN pip install -U torchnet
RUN pip install -U tqdm
