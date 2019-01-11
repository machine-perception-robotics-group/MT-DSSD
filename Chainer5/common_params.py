#! /usr/bin/env python
# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np

# 学習画像のディレクトリ
images_dir = "/home/ryorsk/SSDsegmentation/for_MTDSSD"

# 学習モデル,ロス値の保存ディレクトリ ※頻繁にアクセスがあるのでオンラインストレージ領域にしない
save_model_dir = "/home/ryorsk/SSDsegmentation/models"

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

# ARCクラスラベル (クラス名にはスペース(空白)は禁止)
arc_labels = [
      "Background",             #0
      "Binder",                 #1
      "Balloons",               #2
      "Baby_Wipes",             #3
      "Toilet_Brush",           #4
      "Toothbrushes",           #5
      "Crayons",                #6
      "Salts",                  #7
      "DVD",                    #8
      "Glue_Sticks",            #9
      "Eraser",                 #10
      "Scissors",               #11
      "Green_Book",             #12
      "Socks",                  #13
      "Irish_Spring",           #14
      "Paper_Tape",             #15
      "Touch_Tissues",          #16
      "Knit_Gloves",            #17
      "Laugh_Out_Loud_Jokes",   #18
      "Pencil_Cup",             #19
      "Mini_Marbles",           #20
      "Neoprene_Weight",        #21
      "Wine_Glasses",           #22
      "Water_Bottle",           #23
      "Reynolds_Pie",           #24
      "Reynolds_Wrap",          #25
      "Robots_Everywhere",      #26
      "Duct_Tape",              #27
      "Sponges",                #28
      "Speed_Stick",            #29
      "Index_Cards",            #30
      "Ice_Cube_Tray",          #31
      "Table_Cover",            #32
      "Measuring_Spoons",       #33
      "Bath_Sponge",            #34
      "Pencils",                #35
      "Mousetraps",             #36
      "Face_Cloth",             #37
      "Tennis_Balls",           #38
      "Spray_Bottle",           #39
      "Flashlights"]            #40

# クラスの色
arc_class_color = np.array([
           [  0,   0,   0],
           [ 85,   0,   0],
           [170,   0,   0],
           [255,   0,   0],
           [  0,  85,   0],
           [ 85,  85,   0],
           [170,  85,   0],
           [255,  85,   0],
           [  0, 170,   0],
           [ 85, 170,   0],
           [170, 170,   0],
           [255, 170,   0],
           [  0, 255,   0],
           [ 85, 255,   0],
           [170, 255,   0],
           [255, 255,   0],
           [  0,   0,  85],
           [ 85,   0,  85],
           [170,   0,  85],
           [255,   0,  85],
           [  0,  85,  85],
           [ 85,  85,  85],
           [170,  85,  85],
           [255,  85,  85],
           [  0, 170,  85],
           [ 85, 170,  85],
           [170, 170,  85],
           [255, 170,  85],
           [  0, 255,  85],
           [ 85, 255,  85],
           [170, 255,  85],
           [255, 255,  85],
           [  0,   0, 170],
           [ 85,   0, 170],
           [170,   0, 170],
           [255,   0, 170],
           [  0,  85, 170],
           [ 85,  85, 170],
           [170,  85, 170],
           [255,  85, 170],
           [  0, 170, 170]])
