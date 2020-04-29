#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import datetime
import json
import multiprocessing
import os
import random
import sys
import threading
import time
import math

import numpy as np
import six
import six.moves.cPickle as pickle
from six.moves import queue

import chainer
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import link
import chainer.functions as F

from glob import glob
from os import path

import cPickle
import cv2 as cv

import common_params
from data_augmentation import trainAugmentation
from my_func import SumSquaredError
from my_func import sum_squared_error

def copy_model(src, dst):
    assert isinstance(src, link.Chain)
    assert isinstance(dst, link.Chain)

    for child in src.children():
        if child.name not in dst.__dict__:
            continue
        dst_child = dst[child.name]
        if type(child) != type(dst_child):
            continue
        if isinstance(child, link.Chain):
            copy_model(child, dst_child)
        if isinstance(child, link.Link):
            match = True
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                if a[0] != b[0]:
                    match = False
                    break
                if a[1].data.shape != b[1].data.shape:
                    match = False
                    break
            if not match:
                print('Ignore %s because of parameter mismatch' % child.name)
                continue
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                b[1].data = a[1].data
            print('Copy %s' % child.name)



parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', '-B', type = int, default = 8, help = 'Learning minibatch size')
parser.add_argument('--epoch', '-E', default = 80, type = int, help='Number of epochs to learn')
parser.add_argument('--gpu', '-g', default = -1, type = int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--loaderjob', '-j', default = 4, type=int, help='Number of parallel data loading processes')
parser.add_argument('--suffix', '-S', default = "", type=str, help='Suffix of model saving directory')
args = parser.parse_args()

if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np


print('VGG Netの読み込み中...')
from VGG_Net import VGGNet
vgg_model = VGGNet()
serializers.load_npz('./pretrained_model/VGGNet_for_SSD_ILSVRC.model', vgg_model)
print('-> 読み込み完了')

from SSD_Net import SSDNet
ssd_model = SSDNet()

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    vgg_model.to_gpu()
    ssd_model.to_gpu()

copy_model(vgg_model, ssd_model)

del (vgg_model)

# Setup optimizer
# AdamよりMomentumSGDの方か良い？？ (YOLOはMomentumSGDで実装されている)
#optimizer = optimizers.Adam()
optimizer = optimizers.MomentumSGD(lr = common_params.learning_rate, momentum = common_params.momentum)

optimizer.setup(ssd_model)

optimizer.add_hook(chainer.optimizer.WeightDecay(common_params.weight_decay))

# epoch数
n_epoch = args.epoch

# バッチサイズ
batchsize = args.batchsize

step = int(math.floor((common_params.max_ratio - common_params.min_ratio) / (len(common_params.mbox_source_layers) - 2)))

min_sizes = []
max_sizes = []

# Default boxの最小・最大サイズを計算
for ratio in range(common_params.min_ratio, common_params.max_ratio + 1, step):
    min_sizes.append(common_params.insize * ratio / 100.)
    max_sizes.append(common_params.insize * (ratio + step) / 100.)

min_sizes = [common_params.insize * 10 / 100.] + min_sizes
max_sizes = [common_params.insize * 20 / 100.] + max_sizes



# 学習データの読み込み
def readTrainData(input_name, confing_image):

    aug_p = open(common_params.images_dir + '/train/img_aug_param/' + input_name + '.txt', 'r')

    in_line = aug_p.readline()
    opath = in_line.split(' \n')
    original_img_path = str(opath[0])

    in_line = aug_p.readline()
    augmentation = int(in_line)

    in_line = aug_p.readline()
    part = in_line.split(' ')
    border_pixels = [int(part[0]), int(part[1]), int(part[2]), int(part[3])]

    in_line = aug_p.readline()
    part = in_line.split(' ')
    crop_param = [float(part[0]), float(part[1]), float(part[2]), float(part[3])]

    in_line = aug_p.readline()
    part = in_line.split(' ')
    hsv_param = [float(part[0]), float(part[1]), float(part[2])]

    in_line = aug_p.readline()
    flip_type = int(in_line)


    # 入力画像の読み込み
    color_img = cv.imread(common_params.images_dir + '/train/rgb/' + original_img_path + '.png', cv.IMREAD_COLOR)

    if color_img is None:
        print('画像が読み込めません')
        print(common_params.images_dir + '/train/rgb/' + original_img_path + '.png')
        sys.exit(1)

    # 画像をSSDの入力サイズにリサイズ
    input_img = cv.resize(color_img, (common_params.insize, common_params.insize), interpolation = cv.INTER_CUBIC)

    if augmentation == 1:
        input_img = trainAugmentation(input_img, border_pixels, crop_param, hsv_param, flip_type)

    if confing_image:
        conf_img = input_img.copy()

    # 画像データをfloatに変換
    input_img = input_img.astype(np.float32)

    # 画像の平均値を引く
    input_img -= np.array([103.939, 116.779, 123.68])

    #input_img /= 255.

    # 画像の次元を(高さ，幅，チャンネル数)から(チャンネル数, 高さ，幅)へ転置
    input_img = input_img.transpose(2, 0, 1)

    gt_boxes = []
    df_boxes = []
    indices = []
    classes = []

    idx_tmp = []

    # positiveサンプルの読み込み
    pos_num = 0
    f = open(common_params.images_dir + '/train/positives/' + input_name + '.txt', 'r')
    for rw in f:
        ln = rw.split(' ')
        classes.append(int(ln[1]))
        gt_boxes.append([float(ln[2]), float(ln[3]), float(ln[4]), float(ln[5])])
        df_boxes.append([float(ln[6]), float(ln[7]), float(ln[8]), float(ln[9])])
        indices.append([int(ln[10]), int(ln[11]), int(ln[12]), int(ln[13])])
        pos_num += 1
    f.close()

    # hard negativeサンプルの読み込み (最大でpositiveサンプル数の3倍)
    neg_num = 0
    f = open(common_params.images_dir + '/train/negatives/' + input_name + '.txt', 'r')
    for rw in f:
        ln = rw.split(' ')
        classes.append(int(ln[1]))
        gt_boxes.append([float(ln[2]), float(ln[3]), float(ln[4]), float(ln[5])])
        df_boxes.append([float(ln[2]), float(ln[3]), float(ln[4]), float(ln[5])])
        idx_tmp.append([int(ln[10]), int(ln[11]), int(ln[12]), int(ln[13])])
        neg_num += 1
    f.close()

    hardneg_size = pos_num * 3 if neg_num > (pos_num * 3) else neg_num

    perm = np.random.permutation(len(idx_tmp))

    for hn in range(0, hardneg_size):
        indices.append(idx_tmp[perm[hn]])


    return (input_img, gt_boxes, df_boxes, indices, classes, conf_img)



# 誤差関数
def lossFunction(Loc, Cls, gt_box_batch, df_box_batch, idx_batch, cls_batch, bat_s, mining):

    if mining:
        # hard negative mining有効時のクラスラベル
        cls_t1 = np.ones((bat_s, common_params.num_boxes[0], common_params.map_sizes[0], common_params.map_sizes[0]), np.int32) * -1
        cls_t2 = np.ones((bat_s, common_params.num_boxes[1], common_params.map_sizes[1], common_params.map_sizes[1]), np.int32) * -1
        cls_t3 = np.ones((bat_s, common_params.num_boxes[2], common_params.map_sizes[2], common_params.map_sizes[2]), np.int32) * -1
        cls_t4 = np.ones((bat_s, common_params.num_boxes[3], common_params.map_sizes[3], common_params.map_sizes[3]), np.int32) * -1
        cls_t5 = np.ones((bat_s, common_params.num_boxes[4], common_params.map_sizes[4], common_params.map_sizes[4]), np.int32) * -1
        cls_t6 = np.ones((bat_s, common_params.num_boxes[5], common_params.map_sizes[5], common_params.map_sizes[5]), np.int32) * -1
    else:
        # hard negative mining無効時のクラスラベル
        cls_t1 = np.zeros((bat_s, common_params.num_boxes[0], common_params.map_sizes[0], common_params.map_sizes[0]), np.int32)
        cls_t2 = np.zeros((bat_s, common_params.num_boxes[1], common_params.map_sizes[1], common_params.map_sizes[1]), np.int32)
        cls_t3 = np.zeros((bat_s, common_params.num_boxes[2], common_params.map_sizes[2], common_params.map_sizes[2]), np.int32)
        cls_t4 = np.zeros((bat_s, common_params.num_boxes[3], common_params.map_sizes[3], common_params.map_sizes[3]), np.int32)
        cls_t5 = np.zeros((bat_s, common_params.num_boxes[4], common_params.map_sizes[4], common_params.map_sizes[4]), np.int32)
        cls_t6 = np.zeros((bat_s, common_params.num_boxes[5], common_params.map_sizes[5], common_params.map_sizes[5]), np.int32)

    # bounding boxのオフセットベクトルの教示データ
    loc_t1 = np.zeros((bat_s, common_params.num_boxes[0] * common_params.num_of_offset_dims, common_params.map_sizes[0], common_params.map_sizes[0]), np.float32)
    loc_t2 = np.zeros((bat_s, common_params.num_boxes[1] * common_params.num_of_offset_dims, common_params.map_sizes[1], common_params.map_sizes[1]), np.float32)
    loc_t3 = np.zeros((bat_s, common_params.num_boxes[2] * common_params.num_of_offset_dims, common_params.map_sizes[2], common_params.map_sizes[2]), np.float32)
    loc_t4 = np.zeros((bat_s, common_params.num_boxes[3] * common_params.num_of_offset_dims, common_params.map_sizes[3], common_params.map_sizes[3]), np.float32)
    loc_t5 = np.zeros((bat_s, common_params.num_boxes[4] * common_params.num_of_offset_dims, common_params.map_sizes[4], common_params.map_sizes[4]), np.float32)
    loc_t6 = np.zeros((bat_s, common_params.num_boxes[5] * common_params.num_of_offset_dims, common_params.map_sizes[5], common_params.map_sizes[5]), np.float32)

    for b in range(0, len(idx_batch)):
        for i in range(0, len(idx_batch[b])):

            fmap_layer = idx_batch[b][i][1]
            fmap_position = idx_batch[b][i][2]
            df_box_num = idx_batch[b][i][3]
            st_box_idx = df_box_num * common_params.num_of_offset_dims
            ed_box_idx = st_box_idx + common_params.num_of_offset_dims

            c = int(fmap_position % common_params.map_sizes[fmap_layer])
            r = int(fmap_position / common_params.map_sizes[fmap_layer])

            item_class_id = cls_batch[b][i]

            # 1〜6番目のdefault boxのクラスとオフセットの教示データを格納
            if fmap_layer == 0:
                cls_t1[b, df_box_num, r, c] = item_class_id
                loc_t1[b, st_box_idx : ed_box_idx, r, c] = (np.array(gt_box_batch[b][i], np.float32) - np.array(df_box_batch[b][i], np.float32)) / common_params.loc_var
            elif fmap_layer == 1:
                cls_t2[b, df_box_num, r, c] = item_class_id
                loc_t2[b, st_box_idx : ed_box_idx, r, c] = (np.array(gt_box_batch[b][i], np.float32) - np.array(df_box_batch[b][i], np.float32)) / common_params.loc_var
            elif fmap_layer == 2:
                cls_t3[b, df_box_num, r, c] = item_class_id
                loc_t3[b, st_box_idx : ed_box_idx, r, c] = (np.array(gt_box_batch[b][i], np.float32) - np.array(df_box_batch[b][i], np.float32)) / common_params.loc_var
            elif fmap_layer == 3:
                cls_t4[b, df_box_num, r, c] = item_class_id
                loc_t4[b, st_box_idx : ed_box_idx, r, c] = (np.array(gt_box_batch[b][i], np.float32) - np.array(df_box_batch[b][i], np.float32)) / common_params.loc_var
            elif fmap_layer == 4:
                cls_t5[b, df_box_num, r, c] = item_class_id
                loc_t5[b, st_box_idx : ed_box_idx, r, c] = (np.array(gt_box_batch[b][i], np.float32) - np.array(df_box_batch[b][i], np.float32)) / common_params.loc_var
            elif fmap_layer == 5:
                cls_t6[b, df_box_num, r, c] = item_class_id
                loc_t6[b, st_box_idx : ed_box_idx, r, c] = (np.array(gt_box_batch[b][i], np.float32) - np.array(df_box_batch[b][i], np.float32)) / common_params.loc_var


    # 1〜6階層目の教示confidence mapをint32型のarrayにする
    cls_t1 = xp.array(cls_t1, np.int32)
    cls_t2 = xp.array(cls_t2, np.int32)
    cls_t3 = xp.array(cls_t3, np.int32)
    cls_t4 = xp.array(cls_t4, np.int32)
    cls_t5 = xp.array(cls_t5, np.int32)
    cls_t6 = xp.array(cls_t6, np.int32)

    # 1〜6階層目の教示confidence mapをVariableにする
    cls_t1_data = chainer.Variable(cls_t1)
    cls_t2_data = chainer.Variable(cls_t2)
    cls_t3_data = chainer.Variable(cls_t3)
    cls_t4_data = chainer.Variable(cls_t4)
    cls_t5_data = chainer.Variable(cls_t5)
    cls_t6_data = chainer.Variable(cls_t6)

    # 1〜6階層目の教示confidence mapの次元を(バッチ数, DF box数, 高さ, 幅)から(バッチ数, 高さ, 幅, DF box数)に転置
    cls_t1_data = F.transpose(cls_t1_data, [0, 2, 3, 1])
    cls_t2_data = F.transpose(cls_t2_data, [0, 2, 3, 1])
    cls_t3_data = F.transpose(cls_t3_data, [0, 2, 3, 1])
    cls_t4_data = F.transpose(cls_t4_data, [0, 2, 3, 1])
    cls_t5_data = F.transpose(cls_t5_data, [0, 2, 3, 1])
    cls_t6_data = F.transpose(cls_t6_data, [0, 2, 3, 1])

    # 1〜6階層目の教示confidence mapの各次元数を(バッチ数, 高さ, 幅, DF box数)から(バッチ数 * 高さ * 幅 * DF box数)にreshape
    cls_t1_data = F.reshape(cls_t1_data, [cls_t1_data.data.shape[0] * cls_t1_data.data.shape[1] * cls_t1_data.data.shape[2] * common_params.num_boxes[0]])
    cls_t2_data = F.reshape(cls_t2_data, [cls_t2_data.data.shape[0] * cls_t2_data.data.shape[1] * cls_t2_data.data.shape[2] * common_params.num_boxes[1]])
    cls_t3_data = F.reshape(cls_t3_data, [cls_t3_data.data.shape[0] * cls_t3_data.data.shape[1] * cls_t3_data.data.shape[2] * common_params.num_boxes[2]])
    cls_t4_data = F.reshape(cls_t4_data, [cls_t4_data.data.shape[0] * cls_t4_data.data.shape[1] * cls_t4_data.data.shape[2] * common_params.num_boxes[3]])
    cls_t5_data = F.reshape(cls_t5_data, [cls_t5_data.data.shape[0] * cls_t5_data.data.shape[1] * cls_t5_data.data.shape[2] * common_params.num_boxes[4]])
    cls_t6_data = F.reshape(cls_t6_data, [cls_t6_data.data.shape[0] * cls_t6_data.data.shape[1] * cls_t6_data.data.shape[2] * common_params.num_boxes[5]])

    # 1〜6階層目の教示localization mapをfloat32型のarrayにする
    loc_t1 = xp.array(loc_t1, np.float32)
    loc_t2 = xp.array(loc_t2, np.float32)
    loc_t3 = xp.array(loc_t3, np.float32)
    loc_t4 = xp.array(loc_t4, np.float32)
    loc_t5 = xp.array(loc_t5, np.float32)
    loc_t6 = xp.array(loc_t6, np.float32)

    # 1〜6階層目の教示localization mapをVariableにする
    loc_t1_data = chainer.Variable(loc_t1)
    loc_t2_data = chainer.Variable(loc_t2)
    loc_t3_data = chainer.Variable(loc_t3)
    loc_t4_data = chainer.Variable(loc_t4)
    loc_t5_data = chainer.Variable(loc_t5)
    loc_t6_data = chainer.Variable(loc_t6)

    # 1〜6階層目の教示localization mapの次元を(バッチ数, オフセット次元数 * DF box数, 高さ, 幅)から(バッチ数, 高さ, 幅, オフセット次元数 * DF box数)に転置
    loc_t1_data = F.transpose(loc_t1_data, [0, 2, 3, 1])
    loc_t2_data = F.transpose(loc_t2_data, [0, 2, 3, 1])
    loc_t3_data = F.transpose(loc_t3_data, [0, 2, 3, 1])
    loc_t4_data = F.transpose(loc_t4_data, [0, 2, 3, 1])
    loc_t5_data = F.transpose(loc_t5_data, [0, 2, 3, 1])
    loc_t6_data = F.transpose(loc_t6_data, [0, 2, 3, 1])

    # 1〜6階層目の教示localization mapの各次元数を(バッチ数, 高さ, 幅, オフセット次元数 * DF box数)から(バッチ数 * 高さ * 幅 * DF box数, オフセット次元数)にreshape
    loc_t1_data = F.reshape(loc_t1_data, [loc_t1_data.data.shape[0] * loc_t1_data.data.shape[1] * loc_t1_data.data.shape[2] * common_params.num_boxes[0], int(loc_t1_data.data.shape[3] / common_params.num_boxes[0])])
    loc_t2_data = F.reshape(loc_t2_data, [loc_t2_data.data.shape[0] * loc_t2_data.data.shape[1] * loc_t2_data.data.shape[2] * common_params.num_boxes[1], int(loc_t2_data.data.shape[3] / common_params.num_boxes[1])])
    loc_t3_data = F.reshape(loc_t3_data, [loc_t3_data.data.shape[0] * loc_t3_data.data.shape[1] * loc_t3_data.data.shape[2] * common_params.num_boxes[2], int(loc_t3_data.data.shape[3] / common_params.num_boxes[2])])
    loc_t4_data = F.reshape(loc_t4_data, [loc_t4_data.data.shape[0] * loc_t4_data.data.shape[1] * loc_t4_data.data.shape[2] * common_params.num_boxes[3], int(loc_t4_data.data.shape[3] / common_params.num_boxes[3])])
    loc_t5_data = F.reshape(loc_t5_data, [loc_t5_data.data.shape[0] * loc_t5_data.data.shape[1] * loc_t5_data.data.shape[2] * common_params.num_boxes[4], int(loc_t5_data.data.shape[3] / common_params.num_boxes[4])])
    loc_t6_data = F.reshape(loc_t6_data, [loc_t6_data.data.shape[0] * loc_t6_data.data.shape[1] * loc_t6_data.data.shape[2] * common_params.num_boxes[5], int(loc_t6_data.data.shape[3] / common_params.num_boxes[5])])

    # 1〜6階層目の教示confidence mapを結合
    Cls_T = F.concat([cls_t1_data, cls_t2_data, cls_t3_data, cls_t4_data, cls_t5_data, cls_t6_data], axis = 0)


    # 1〜6階層目の教示localization mapを結合
    Loc_T = F.concat([loc_t1_data, loc_t2_data, loc_t3_data, loc_t4_data, loc_t5_data, loc_t6_data], axis = 0)

    # confidence mapのloss
    loss_cls = F.softmax_cross_entropy(Cls, Cls_T)

    # localization mapのloss
    # オリジナル実装ではSmooth L1 Lossだが、本ソースコードではMean Squared Errorを使用
    # Smooth L1 Lossで誤差を求める場合は「F.huber_loss(x, t, delta = 1.0)」
    loss_loc = F.mean_squared_error(Loc, Loc_T)
    #loss_loc = F.mean_absolute_error(Loc, LT)
    #loss_loc = sum_squared_error(Loc, LT)

    del cls_t1, cls_t2, cls_t3, cls_t4, cls_t5, cls_t6
    del loc_t1, loc_t2, loc_t3, loc_t4, loc_t5, loc_t6
    del cls_t1_data, cls_t2_data, cls_t3_data, cls_t4_data, cls_t5_data, cls_t6_data
    del loc_t1_data, loc_t2_data, loc_t3_data, loc_t4_data, loc_t5_data, loc_t6_data

    return (0.4 * loss_loc) + (0.6 * loss_cls)


# 学習データリストの読み込み
f = open('./augimg_name_list.txt', 'r')
input_list = []
for line in f:
    ln = line.split('\n')
    input_list.append(ln[0])
f.close()

input_list = np.array(input_list)

#　学習データ数
N = len(input_list)
print ('Training samples : ', N)
itr_loss_save = int((float(N) / float(batchsize)) * 0.1) * 10
print ('Training samples per batchsize : ', float(N) / float(batchsize))
print ('Iteration save : ', itr_loss_save)

ssd_model.train = True

today = datetime.datetime.today()

today_dir = str(today.year) + '-' + str('%02d' % today.month) + '-' + str('%02d' % today.day) + '@' + str('%02d' % today.hour) + '-' + str('%02d' % today.minute) + '-' + str('%02d' % today.second) + '_' + args.suffix

today_dir_path = common_params.save_model_dir + '/' + today_dir

if not path.exists(today_dir_path):
    os.mkdir(today_dir_path)

fout = open(today_dir_path + '/loss.txt', 'w')

save_model_path = today_dir_path + '/model'
save_optimizer_path = today_dir_path + '/optimizer'

if not path.exists(save_model_path):
    os.mkdir(save_model_path)

if not path.exists(save_optimizer_path):
    os.mkdir(save_optimizer_path)


data_q = queue.Queue(maxsize=1)
res_q = queue.Queue()

def feed_data():
    i = 0

    batch_pool = [None] * batchsize

    pool = multiprocessing.Pool(args.loaderjob)

    data_q.put('train')

    # エポックループ
    for epoch in range(0, n_epoch):

        # changing learning rate (common_params.learning_rate * (common_params.lr_step ** (n_epoch - common_params.lr_change_epoch)))
        if (epoch + 1) >= common_params.lr_change_epoch:
            optimizer.lr *= common_params.lr_step if optimizer.lr >= common_params.lower_lr else common_params.lower_lr


        print ('\nEpoch: %d, Learning rate: %f (%s)' % (epoch + 1, optimizer.lr, today_dir))

        perm = np.random.permutation(N)

        # 学習サンプルのループ
        for dt in range(0, N):

            x_list = input_list[perm[dt]]

            batch_pool[i] = pool.apply_async(readTrainData, (x_list, True))
            i += 1

            img_batch = []
            gt_box_batch = []
            df_box_batch = []
            idx_batch = []
            cls_batch = []
            conf_img_batch = []

            if i == batchsize:
                for inc, x_data in enumerate(batch_pool):
                    input_img, gt_boxes, df_boxes, indices, classes, conf_img = x_data.get()
                    img_batch.append(input_img)
                    gt_box_batch.append(gt_boxes)
                    df_box_batch.append(df_boxes)
                    idx_batch.append(indices)
                    cls_batch.append(classes)
                    conf_img_batch.append(conf_img)
                data_q.put((img_batch, gt_box_batch, df_box_batch, idx_batch, cls_batch, conf_img_batch, epoch + 1))
                i = 0

            del img_batch, gt_box_batch, df_box_batch, idx_batch, cls_batch, conf_img_batch

        # 学習したモデルとoptimizerを保存
        if epoch + 1 >= 50:
            if (epoch + 1) % 2 == 0:
                serializers.save_npz(save_model_path + "/SSD_epoch_" + str(epoch + 1) + "_with_mining.model", ssd_model)
                serializers.save_npz(save_optimizer_path + "/SSD_epoch_" + str(epoch + 1) + "_with_mining.state", optimizer)
            else:
                serializers.save_npz(save_model_path + "/SSD_epoch_" + str(epoch + 1) + "_without_mining.model", ssd_model)
                serializers.save_npz(save_optimizer_path + "/SSD_epoch_" + str(epoch + 1) + "_without_mining.state", optimizer)

    pool.close()
    pool.join()
    data_q.put('end')

def log_result():

    sum_loss = 0.
    sum_bat = 0
    itr = 0
    begin_at = time.time()

    while True:
        result = res_q.get()
        if result == 'end':
            break
        elif result == 'train':
            continue

        loss, bat = result

        duration = time.time() - begin_at

        sum_loss += loss
        sum_bat += bat
        itr += 1

        sys.stdout.write('\rUpdates: {}, Time: {}, Loss: {}'.format(itr, datetime.timedelta(seconds = duration), loss))
        sys.stdout.flush()
        time.sleep(0.01)

        if itr % itr_loss_save == 0:
            print (" [%d iteration loss: %f]" % (itr, sum_loss / float(itr_loss_save)))
            out_line = '{} {} {} \n'.format(itr, sum_loss / float(itr_loss_save), optimizer.lr)
            fout.write(out_line)
            sum_loss = 0.
            sum_bat = 0


def train_loop():

    while True:
        input_data = data_q.get()
        if input_data == 'end':
            res_q.put('end')
            break
        elif input_data == 'train':
            res_q.put('train')
            continue

        img_batch, gt_box_batch, df_box_batch, idx_batch, cls_batch, conf_img_batch, epoch_num = input_data

        # ---教師ラベル確認用(画像の確認が不要な場合は以下14行をコメントアウト)----------------------------------
        # for b in range(0, len(conf_img_batch)):
        #     font = cv.FONT_HERSHEY_SIMPLEX
        #     for bx in range(len(gt_box_batch[b]) - 1, 0, -1):
        #       p1 = int(gt_box_batch[b][bx][0] * common_params.insize)
        #       p2 = int(gt_box_batch[b][bx][1] * common_params.insize)
        #       p3 = int(gt_box_batch[b][bx][2] * common_params.insize)
        #       p4 = int(gt_box_batch[b][bx][3] * common_params.insize)
        #       cv.rectangle(conf_img_batch[b], (p1, p2), (p3, p4), (0, 255, 0), 2)
        #       q1 = p1
        #       q2 = p4
        #       cv.rectangle(conf_img_batch[b], (q1, q2 - 25), (q1 + 25, q2), (0, 255, 0), -1)
        #       cv.putText(conf_img_batch[b], str(cls_batch[b][bx]), (q1, q2 - 8), font, 0.6, (0, 0, 0), 1, cv.CV_AA)
        #     cv.imshow('Augmentation', conf_img_batch[b])
        #     cv.waitKey()
        # ------------------------------------------------------------------------------------------

        bat_s = len(img_batch)

        ssd_model.cleargrads()

        train_img = chainer.Variable(xp.array(img_batch))

        # SSD net forward
        Loc1, Cls1, Loc2, Cls2, Loc3, Cls3, Loc4, Cls4, Loc5, Cls5, Loc6, Cls6 = ssd_model(train_img)

        # ネットワークから出力されたconfidence mapを1〜6階層目まで結合
        Loc = F.concat([Loc1, Loc2, Loc3, Loc4, Loc5, Loc6], axis = 0)

        # ネットワークから出力されたlocalization mapを1〜6階層目まで結合
        Cls = F.concat([Cls1, Cls2, Cls3, Cls4, Cls5, Cls6], axis = 0)

        # epochが2回に1回の割合でhard negative mining有効
        if epoch_num % 2 == 0:
            mining = True
        else:
            mining = False

        # lossを計算
        loss = lossFunction(Loc, Cls, gt_box_batch, df_box_batch, idx_batch, cls_batch, bat_s, mining)
        loss.backward()

        optimizer.update()

        res_q.put((float(loss.data), float(bat_s)))

        del img_batch, gt_box_batch, df_box_batch, idx_batch, cls_batch, conf_img_batch

        del train_img
        del Loc1, Loc2, Loc3, Loc4, Loc5, Loc6
        del Cls1, Cls2, Cls3, Cls4, Cls5, Cls6
        del Loc, Cls



feeder = threading.Thread(target = feed_data)
feeder.daemon = True
feeder.start()
logger = threading.Thread(target = log_result)
logger.daemon = True
logger.start()

train_loop()
feeder.join()
logger.join()

fout.close()
print("\nExit Training\n")
