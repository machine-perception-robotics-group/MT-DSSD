FROM python:3
FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04

ENV PATH /usr/lib/ccache:$PATH

RUN apt-get -y update && apt-get -y upgrade && \
  apt-get install -y ccache curl g++ gfortran git libhdf5-dev libopencv-dev && \
  apt-get clean
RUN apt-get install -y python3-pip python3-dev && \
  apt-get clean

RUN apt-get install -y language-pack-ja-base language-pack-ja
  echo 'export LANG=ja_JP.UTF-8' >> ~/.bashrc
  echo 'export LC_CTYPE=ja_JP.UTF-8' >> ~/.bashrc

RUN echo 'alias python=python3' >> ~/.bashrc
RUN echo 'alias pip=pip3' >> ~/.bashrc

RUN apt-get -y -q install python-opencv && \
    apt-get -y -q install python-tk && \
    apt-get -y install vim && \
    apt-get -y install wget

RUN pip3 install --upgrade pip
ENV CUDA_ROOT /usr/local/cuda
ENV PATH $PATH:$CUDA_ROOT/bin
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$CUDA_ROOT/lib64:$CUDA_ROOT/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib
ENV LIBRARY_PATH /usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64:/usr/local/cuda/lib$LIBRARY_PATH

RUN pip3 install -U setuptools
RUN pip3 install -U cython
RUN pip3 install -U h5py
RUN pip3 install -U scipy
RUN pip3 install -U matplotlib
RUN pip3 install -U hacking
RUN pip3 install -U nose
RUN pip3 install -U mock
RUN pip3 install -U coverage
RUN pip3 install -U scikit-learn
RUN pip3 install -U pillow
RUN pip3 install -U opencv-python
RUN pip3 install -U chainer
RUN pip3 install -U cupy
