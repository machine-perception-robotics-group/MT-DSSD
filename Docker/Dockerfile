FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

ENV PATH /usr/lib/ccache:$PATH

RUN apt-get -y update && apt-get -y upgrade && \
  apt-get install -y ccache curl g++ gfortran git libhdf5-dev libopencv-dev && \
  apt-get clean
RUN apt-get install -y python-pip python-dev && \
  apt-get clean

RUN apt-get install -y language-pack-ja-base language-pack-ja && \
  echo 'export LANG=ja_JP.UTF-8' >> ~/.bashrc && \
  echo 'export LC_CTYPE=ja_JP.UTF-8' >> ~/.bashrc

RUN apt-get -y -q install python-tk && \
    apt-get -y install vim && \
    apt-get -y install wget

RUN pip install --upgrade pip
ENV CUDA_ROOT /usr/local/cuda
ENV PATH $PATH:$CUDA_ROOT/bin
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$CUDA_ROOT/lib64:$CUDA_ROOT/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib
ENV LIBRARY_PATH /usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64:/usr/local/cuda/lib$LIBRARY_PATH

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
RUN pip install -U cupy==2.1.0
RUN pip install -U chainer==1.24.0
