# MT-DSSD (with Chainer5)
MT-DSSDのChainer5実装版（作成中）です．
loss.backward()時にcuDNNが使えないため，cuDNNを使わずに学習してください．
他にもバグがあるかもしれません．

## Contents
- SSD[1] (準備中)
- MultiTask DSSD(MT-DSSD)[2, 3]

# Requirement
- Python 2.x (Recommended version >= 2.7.11)
- OpenCV 2.x
- Chainer 5.1
- Cupy 5.1
- numpy (Recommended version >= 1.10)
- tqdm

# Usage
Chainer1版を参照してください．

Docker実行例：
```
sudo nvidia-docker build ./ -t cuda92_chainer5_1:latest
sudo nvidia-docker run --shm-size 8G -it -v /home/ryorsk:/home/ryorsk cuda92_chainer5_1:latest
```

# Reference
- [1] W. Liu, et al., “SSD: Single Shot MultiBox Detector”, ECCV, pp. 21–37, 2016.
- [2] C. Fu, et al., “DSSD: Deconvolutional Single Shot Detector”, arXiv preprint arXiv:1701.06659, 2017.
- [3] 荒木, et al., “マルチタスク学習を導入したDeconvolutional Single Shot Detectorによる物体検出とセグメンテーションの高精度化”, MIRU, 2018.
