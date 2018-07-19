# -*- coding: utf-8 -*-
import cv2 as cv

# 学習画像のディレクトリ
images_dir = '/home/ryorsk/SSDsegmentation/for_MTDSSD'

# 学習モデル,ロス値の保存ディレクトリ ※頻繁にアクセスがあるのでオンラインストレージ領域にしない
save_model_dir = './models'

# SSDの入力画像サイズ
insize = 300

# 識別のクラス数 (背景込み)
num_of_classes = 41
#ARC:41, VOC:21

# Bounding boxのオフセットベクトルの次元数
num_of_offset_dims = 4

# Bounding boxのオフセットとクラスを推定する畳み込み層
mbox_source_layers = ['conv9_2', 'preconv3_conv8_2', 'esconv2', 'esconv3', 'esconv4', 'esconv5']

# Default boxの最小・最大比率 (in percent %)
min_ratio = 20
max_ratio = 90

# 各階層における特徴マップの入力画像上のステップ幅
steps = [8, 16, 32, 64, 100, 300]

# 各階層のdefault boxの数
num_boxes = [4, 6, 6, 6, 4, 4]

# 各階層の特徴マップのDefault boxのアスペクト比
aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

# 各階層の特徴マップの解像度
map_sizes = [38, 19, 10, 5, 3, 1]

loc_var = 0.01

# Data Augmentationで増加させる倍数
augmentation_factor = 30
#ARC:30倍，VOC:15倍

# Data Augmentationのパラメータ
jitter = 0.3      # Default : 0.2
saturation = 1.5  # Default : 1.5
exposure = 1.5    # Default : 1.5
hue = 0.03        # Default : 0.03

# choose border type
# BORDER_REPLICATE, BORDER_REFLECT, BORDER_REFLECT_101, BORDER_WRAP, BORDER_CONSTANT
border_type = cv.BORDER_CONSTANT

#border_val = (36, 36, 71)
border_val = (127, 127, 127)
#border_val = (0, 0, 0)

# optimizer (MomentumSGDの学習パラメータ)
learning_rate = 10e-3
momentum = 0.9
weight_decay = 0.0005

lr_step = 0.97
lr_change_epoch = 80
lower_lr = 10e-4
