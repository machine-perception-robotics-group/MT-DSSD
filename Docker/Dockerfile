FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ENV PATH /usr/lib/ccache:$PATH
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get -y upgrade && \
  apt-get install -y ccache curl g++ gfortran git libhdf5-dev libopencv-dev vim wget python3-pip python3-dev python3-tk graphviz && \
  apt-get clean

RUN apt-get install -y language-pack-ja-base language-pack-ja && \
  echo 'export LANG=ja_JP.UTF-8' >> ~/.bashrc && \
  echo 'export LC_CTYPE=ja_JP.UTF-8' >> ~/.bashrc

RUN pip3 install --upgrade pip
ENV CUDA_ROOT /usr/local/cuda
ENV PATH $PATH:$CUDA_ROOT/bin
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$CUDA_ROOT/lib64:$CUDA_ROOT/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib
ENV LIBRARY_PATH /usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64:/usr/local/cuda/lib$LIBRARY_PATH

RUN pip3 install -U setuptools cython h5py scipy matplotlib hacking nose mock coverage scikit-learn pillow opencv-python==3.4.10.35 cupy-cuda100==5.1.0 chainer==5.1.0 numba tqdm
