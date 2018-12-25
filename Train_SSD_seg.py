#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import datetime
import os
import random
import sys
import time
import math
from glob import glob
from os import path

import numpy as np
import six
import six.moves.cPickle as pickle
import cPickle
import cv2 as cv

import chainer
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import link
from chainer import datasets
from chainer import training
from chainer.training import extensions
from chainer.training import triggers
import chainer.functions as F

import common_params
from data_augmentation_seg import trainAugmentation
from my_func import SumSquaredError
from my_func import sum_squared_error

from SSD_seg_Net import SSDNet
from VGG_Net import VGGNet

"""
未実装一覧
・hard neg mining時に保存するモデルの名前を変える
・モデルのロス保存が怪しい
・4epochごとのhard neg mining
・サンプルによってindicesの要素数がゼロになって止まる
"""

parser = argparse.ArgumentParser()  #パーサーを作る
parser.add_argument('--batchsize', '-B', type = int, default = 5, help = 'Learning minibatch size') #引数の追加
parser.add_argument('--epoch', '-E', default = 150, type = int, help='Number of epochs to learn')    #引数の追加
parser.add_argument('--gpu', '-g', default = 0, type = int, help='GPU ID (negative value indicates CPU)')  #引数の追加
parser.add_argument('--loaderjob', '-j', default = 4, type=int, help='Number of parallel data loading processes')   #引数の追加
parser.add_argument('--segmentignore', '-s', default = 0, type=int, help='Even if segmentation image is not found, continue learning (1:True/0:False)')   #引数の追加
parser.add_argument('--resume')
args = parser.parse_args()  #引数を解析

#GPUが使えるか確認
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

#lossの保存間隔
ITR_LOSS_SAVE = 1000

# epoch数
n_epoch = args.epoch

# バッチサイズ
batchsize = args.batchsize

def print_array(name, array):
    print(name)
    print(len(array))
    print(array)
    print()

# for hard neg mining (未確認)
def set_epoch_num(model, epoch):
    @training.make_extension(trigger=(1, 'epoch'))
    def _set_epoch_num(trainer):
        print("set epoch num")
        model.set_epoch_num(epoch)
    return _set_epoch_num

class TrainChain(chainer.Chain):
    def __init__(self, model, today_dir_path):
        super(TrainChain, self).__init__()
        with self.init_scope():
            self.model = model
        self.today_dir_path = today_dir_path
        self.epoch_num = 0

    def __call__(self, img_batch, gt_box_batch, df_box_batch, idx_batch, cls_batch, conf_img_batch, seglabel_batch):
        print("-----------START FORWARD-----------------------------------------")
        bat_s = args.batchsize
        #train_img = chainer.Variable(xp.array(img_batch))
        #seg_label = chainer.Variable(xp.array(seglabel_batch, np.int32))
        #train_img = chainer.Variable(img_batch)
        #seg_label = chainer.Variable(seglabel_batch)
        today_dir_path = self.today_dir_path
        train_img = img_batch
        seg_label = seglabel_batch

        # SSD net forward
        Loc1, Cls1, Loc2, Cls2, Loc3, Cls3, Loc4, Cls4, Loc5, Cls5, Loc6, Cls6, Seg = self.model(train_img)

        # ネットワークから出力されたconfidence mapを1〜6階層目まで結合
        Loc = F.concat([Loc1, Loc2, Loc3, Loc4, Loc5, Loc6], axis = 0)

        # ネットワークから出力されたlocalization mapを1〜6階層目まで結合
        Cls = F.concat([Cls1, Cls2, Cls3, Cls4, Cls5, Cls6], axis = 0)

        epoch_num = self.epoch_num #動作未確認
        # epochが4回に1回の割合でhard negative mining有効
        if epoch_num % 4 == 0:
            mining = True
        else:
            mining = False

        # lossを計算
        print("-----------CALC LOSS-----------------------------------------")
        loss_cls, loss_loc, loss_seg = loss_function_new(Loc, Cls, Seg, gt_box_batch, df_box_batch, idx_batch, cls_batch, bat_s, mining, seg_label, today_dir_path)
        loss = (loss_cls + loss_loc + loss_seg) / 3.
        print(loss)
        print(type(loss))

        chainer.reporter.report(
            {'loss': loss, 'loss/cls': loss_cls, 'loss/loc': loss_loc, 'loss/seg': loss_seg},
            self)
        print("-----------CALC LOSS END-------------------------------------")
        return loss

    def set_epoch_num(self, epoch_num):
        self.epoch_num = epoch_num


class Dataset(chainer.dataset.DatasetMixin):
    def __init__(self, input_list_path, input_seglabel_list_path, confing_image):
        # Open image datalist
        f = open(input_list_path, 'r')
        input_list = []
        for line in f:
            ln = line.split('\n')
            input_list.append(ln[0])
        input_list.sort()
        f.close()

        # Open segmentation label datalist
        f = open(input_seglabel_list_path, 'r')
        input_seglabel_list = []
        for line in f:
            ln = line.split('\n')
            input_seglabel_list.append(ln[0])
        input_seglabel_list.sort()
        f.close()

        self.input_list = np.array(input_list)
        self.input_seglabel_list = np.array(input_seglabel_list)
        self.confing_image = confing_image
        #perm = np.random.permutation(N) #学習サンプルのシャッフル

        print("Training images : ", len(self.input_list))
        print("Segmentation labels : ", len(self.input_seglabel_list))
        if len(self.input_list) != len(self.input_seglabel_list):
            print("[ERROR] Mismatch between #input_images and #segmentation_labels.")
            exit(1)

    def __len__(self):
        return len(self.input_list)

    def get_example(self, i):
        input_name = self.input_list[i]
        input_seglabel_name = self.input_seglabel_list[i]
        #
        # Detection teach
        #

        # Augmentation Params
        aug_p = open(common_params.images_dir + "/train/img_aug_param/" + input_name + ".txt", 'r')

        in_line = aug_p.readline()  #aug_pを1行ずつ読み込み
        opath = in_line.split(' \n')    #改行で区切る
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

        #
        # Segmentation teach
        #

        # Segmentation Aug Param (Actually, segmentation image path)
        seg_label_p = open(common_params.images_dir + "/train/seglabel_aug_param/" + input_seglabel_name + ".txt", 'r')

        in_line = seg_label_p.readline()
        slpath = in_line.split(' \n')
        seg_label_path = str(slpath[0])

        # Input image
        color_img = cv.imread(common_params.images_dir + "/train/rgb/" + original_img_path + ".png", cv.IMREAD_COLOR)

        if color_img is None:
            print("[ERROR] Cannot read input image:")
            print(common_params.images_dir + "/train/rgb/" + original_img_path + ".png")
            sys.exit(1)

        # セグメンテーション画像の読み込み
        seg_labelimg = cv.imread(common_params.images_dir + "/train/seglabel/" + seg_label_path + ".png", 0)

        if seg_labelimg is None:
            if args.segmentignore == 1:
                seg_labelimg = np.ones((common_params.insize, common_params.insize)) * -1
            else:
                print("[ERROR] Cannot read segmentation label")
                print(common_params.images_dir + "/train/seglabel/" + seg_label_path + ".png")
                print("引数 -segmentignore 1を与えれば無視してB-Boxのみで学習します(Segmentation誤差を伝播させません)")
                sys.exit(1)

        # 画像をSSDの入力サイズにリサイズ
        input_img = cv.resize(color_img, (common_params.insize, common_params.insize), interpolation = cv.INTER_CUBIC)  #バイキュビック補間
        #input_seglabel = np.ones((common_params.insize, common_params.insize)) * -1

        #print("open:" + str(np.max(seg_labelimg[seg_labelimg!=255]))) #デバッグ用 seg教師信号の値確認 255以外で一番でかいやつを出力

        input_seglabel = cv.resize(seg_labelimg, (common_params.insize, common_params.insize), interpolation = cv.INTER_NEAREST)
        #print("resize:" + str(np.max(input_seglabel[input_seglabel!=255]))) #デバッグ用 seg教師信号の値確認 255以外で一番でかいやつを出力

        if augmentation == 1:
            input_img, input_seglabel = trainAugmentation(input_img, border_pixels, crop_param, hsv_param, flip_type, input_seglabel)   #data augmentation
        #print("augmentation:" + str(np.max(input_seglabel[input_seglabel!=255]))) #デバッグ用 seg教師信号の値確認 255以外で一番でかいやつを出力

        if self.confing_image:
            conf_img = input_img.copy()
        #print(input_seglabel)
        #input_seglabel = np.zeros([batchsize, common_params.insize, common_params.insize], dtype=np.int32)

        # 画像データをfloatに変換
        input_img = input_img.astype(np.float32)

        # 画像の平均値を引く
        input_img -= np.array([103.939, 116.779, 123.68])

        # 画像の次元を(高さ，幅，チャンネル数)から(チャンネル数, 高さ，幅)へ転置
        input_img = input_img.transpose(2, 0, 1)

        gt_boxes = []
        df_boxes = []
        indices = []
        classes = []
        idx_tmp = []

        # positiveサンプルの読み込み
        pos_num = 0
        f = open(common_params.images_dir + "/train/positives/" + input_name + ".txt", 'r')
        for rw in f:
            ln = rw.split(' ')
            classes.append(int(ln[1]))
            gt_boxes.append([float(ln[2]), float(ln[3]), float(ln[4]), float(ln[5])])
            df_boxes.append([float(ln[6]), float(ln[7]), float(ln[8]), float(ln[9])])
            indices.append([int(ln[10]), int(ln[11]), int(ln[12]), int(ln[13])])
            print("indices.append", [int(ln[10]), int(ln[11]), int(ln[12]), int(ln[13])])
            pos_num += 1
        f.close()

        # hard negativeサンプルの読み込み (最大でpositiveサンプル数の3倍)
        neg_num = 0
        f = open(common_params.images_dir + "/train/negatives/" + input_name + ".txt", 'r')
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
            print("indices.append", idx_tmp[perm[hn]])

        # segmentationのignore class(255)を-1にする
        input_seglabel = input_seglabel.astype(np.int32) #uintからintにしないと負の値が入らない
        input_seglabel[input_seglabel==255] = -1

        # list to numpy array
        yp = np
        input_img = yp.array(input_img)
        gt_boxes = yp.array(gt_boxes).astype(np.float32)
        df_boxes = yp.array(df_boxes).astype(np.float32)
        indices = yp.array(indices).astype(np.int32)
        classes = yp.array(classes).astype(np.int32)
        conf_img = yp.array(conf_img)
        input_seglabel = yp.array(input_seglabel)

        print("indices", indices, indices.shape)
        # padding ???(random) -> 8732(dfbox max size)
        if len(gt_boxes) != 8732:
            gt_boxes = np.pad(gt_boxes, [(0,8732-len(gt_boxes)), (0,0)], 'constant')
            df_boxes = np.pad(df_boxes, [(0,8732-len(df_boxes)), (0,0)], 'constant')
            indices = np.pad(indices, [(0,8732-len(indices)), (0,0)], 'constant')
            classes = np.pad(classes, (0,8732-len(classes)), 'constant')

        #print("final:" + str(np.max(input_seglabel))) #デバッグ用 seg教師信号の値確認 255以外で一番でかいやつを出力
        """
        print_array("input_img", input_img)
        print_array("gt_boxes", gt_boxes)
        print_array("df_boxes", df_boxes)
        print_array("indices", indices)
        print_array("classes", classes)
        print_array("conf_img", conf_img)
        print_array("input_seglabel", input_seglabel)
        """
        """
        print("input_img", np.array(input_img).shape, type(input_img[0][0][0]))
        print("gt_boxes", np.array(gt_boxes).shape, type(gt_boxes[0][0]))
        print("df_boxes", np.array(df_boxes).shape, type(df_boxes[0][0]))
        print("indices", np.array(indices).shape, type(indices[0][0]))
        print("classes", np.array(classes).shape, type(classes[0]))
        print("conf_img", np.array(conf_img).shape, type(conf_img[0][0][0]))
        print("input_seglabel", np.array(input_seglabel).shape, type(input_seglabel[0][0]))
        print("-----------------")
        """
        """
        print("input_img", input_img.shape, type(input_img[0][0][0]))
        print("gt_boxes", gt_boxes.shape, type(gt_boxes[0][0]))
        print("df_boxes", df_boxes.shape, type(df_boxes[0][0]))
        print("indices", indices.shape, type(indices[0][0]))
        print("classes", classes.shape, type(classes[0]))
        print("conf_img", conf_img.shape, type(conf_img[0][0][0]))
        print("input_seglabel", input_seglabel.shape, type(input_seglabel[0][0]))
        print("-----------------")
        """
        return input_img, gt_boxes, df_boxes, indices, classes, conf_img, input_seglabel
        """
        This Code:
        input_img (3, 300, 300) <type 'numpy.float32'>
        gt_boxes (2810, 4) <type 'numpy.float64'>
        df_boxes (2810, 4) <type 'numpy.float64'>
        indices (356, 4) <type 'numpy.int64'>
        classes (2810,) <type 'numpy.int64'>
        conf_img (300, 300, 3) <type 'numpy.uint8'>
        input_seglabel (300, 300) <type 'numpy.int64'>

        ChainerCV:
        img
            shape= (3, 300, 300)
        mb_loc:
            shape= (8732, 4) np.float32
        mb_label:
            shape= (8732,) np.int32

        """


def copy_model(src, dst):   #src:vgg_model dst:ssd_model
    assert isinstance(src, link.Chain)  #assert:条件式がfalseの場合、AssertionErrorが発生　isinstance:一つ目にオブジェクト、二つ目にクラスを受け取り、一つ目に渡したオブジェクトが二つ目のクラスのインスタンスならTrueを返す
    assert isinstance(dst, link.Chain)

    for child in src.children():
        if child.name not in dst.__dict__:  #__dict__リストオブジェクトの中にchild.nameが含まれていればfalse、含まれていなければtreuを返す __dict__:任意の関数属性をサポートするための名前空間が収められている
            continue
        dst_child = dst[child.name]
        if type(child) != type(dst_child):  #type:引数に渡されたオブジェクトのクラスを返す
            continue
        if isinstance(child, link.Chain):
            copy_model(child, dst_child)
        if isinstance(child, link.Link):
            match = True
            for a, b in zip(child.namedparams(), dst_child.namedparams()):  #zip:複数シーケンスオブジェクトを同時にループ
                if a[0] != b[0]:
                    match = False
                    break
                if a[1].data.shape != b[1].data.shape:
                    match = False
                    break
            if not match:
                print("Ignore %s because of parameter mismatch" % child.name)
                continue
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                b[1].data = a[1].data
            print("Copy %s" % child.name)


# 誤差関数(new)
def loss_function_new(Loc, Cls, Seg, gt_box_batch, df_box_batch, idx_batch, cls_batch, bat_s, mining, seg_label, today_dir_path):

    if mining:
        # hard negative mining有効時のクラスラベル
        cls_t1 = xp.ones((bat_s, common_params.num_boxes[0], common_params.map_sizes[0], common_params.map_sizes[0]), np.int32) * -1
        cls_t2 = xp.ones((bat_s, common_params.num_boxes[1], common_params.map_sizes[1], common_params.map_sizes[1]), np.int32) * -1
        cls_t3 = xp.ones((bat_s, common_params.num_boxes[2], common_params.map_sizes[2], common_params.map_sizes[2]), np.int32) * -1
        cls_t4 = xp.ones((bat_s, common_params.num_boxes[3], common_params.map_sizes[3], common_params.map_sizes[3]), np.int32) * -1
        cls_t5 = xp.ones((bat_s, common_params.num_boxes[4], common_params.map_sizes[4], common_params.map_sizes[4]), np.int32) * -1
        cls_t6 = xp.ones((bat_s, common_params.num_boxes[5], common_params.map_sizes[5], common_params.map_sizes[5]), np.int32) * -1
    else:
        # hard negative mining無効時のクラスラベル
        cls_t1 = xp.zeros((bat_s, common_params.num_boxes[0], common_params.map_sizes[0], common_params.map_sizes[0]), np.int32)
        cls_t2 = xp.zeros((bat_s, common_params.num_boxes[1], common_params.map_sizes[1], common_params.map_sizes[1]), np.int32)
        cls_t3 = xp.zeros((bat_s, common_params.num_boxes[2], common_params.map_sizes[2], common_params.map_sizes[2]), np.int32)
        cls_t4 = xp.zeros((bat_s, common_params.num_boxes[3], common_params.map_sizes[3], common_params.map_sizes[3]), np.int32)
        cls_t5 = xp.zeros((bat_s, common_params.num_boxes[4], common_params.map_sizes[4], common_params.map_sizes[4]), np.int32)
        cls_t6 = xp.zeros((bat_s, common_params.num_boxes[5], common_params.map_sizes[5], common_params.map_sizes[5]), np.int32)

    # bounding boxのオフセットベクトルの教示データ
    loc_t1 = xp.zeros((bat_s, common_params.num_boxes[0] * common_params.num_of_offset_dims, common_params.map_sizes[0], common_params.map_sizes[0]), np.float32)
    loc_t2 = xp.zeros((bat_s, common_params.num_boxes[1] * common_params.num_of_offset_dims, common_params.map_sizes[1], common_params.map_sizes[1]), np.float32)
    loc_t3 = xp.zeros((bat_s, common_params.num_boxes[2] * common_params.num_of_offset_dims, common_params.map_sizes[2], common_params.map_sizes[2]), np.float32)
    loc_t4 = xp.zeros((bat_s, common_params.num_boxes[3] * common_params.num_of_offset_dims, common_params.map_sizes[3], common_params.map_sizes[3]), np.float32)
    loc_t5 = xp.zeros((bat_s, common_params.num_boxes[4] * common_params.num_of_offset_dims, common_params.map_sizes[4], common_params.map_sizes[4]), np.float32)
    loc_t6 = xp.zeros((bat_s, common_params.num_boxes[5] * common_params.num_of_offset_dims, common_params.map_sizes[5], common_params.map_sizes[5]), np.float32)

    for b in range(0, len(idx_batch)):
        for i in range(0, len(idx_batch[b])):

            fmap_layer = int(idx_batch[b][i][1])
            fmap_position = int(idx_batch[b][i][2])
            df_box_num = int(idx_batch[b][i][3])
            st_box_idx = df_box_num * common_params.num_of_offset_dims
            ed_box_idx = st_box_idx + common_params.num_of_offset_dims
            c = int(fmap_position % common_params.map_sizes[fmap_layer])
            r = int(fmap_position / common_params.map_sizes[fmap_layer])

            # 1〜6番目のdefault boxのクラスとオフセットの教示データを格納
            gt_box_batch_idx = gt_box_batch[b][i]
            df_box_batch_idx = df_box_batch[b][i]

            if fmap_layer == 0:
                cls_t1[b, df_box_num, r, c] = cls_batch[b][i]
                loc_t1[b, st_box_idx : ed_box_idx, r, c] = (gt_box_batch_idx - df_box_batch_idx) / common_params.loc_var
            elif fmap_layer == 1:
                cls_t2[b, df_box_num, r, c] = cls_batch[b][i]
                loc_t2[b, st_box_idx : ed_box_idx, r, c] = (gt_box_batch_idx - df_box_batch_idx) / common_params.loc_var
            elif fmap_layer == 2:
                cls_t3[b, df_box_num, r, c] = cls_batch[b][i]
                loc_t3[b, st_box_idx : ed_box_idx, r, c] = (gt_box_batch_idx - df_box_batch_idx) / common_params.loc_var
            elif fmap_layer == 3:
                cls_t4[b, df_box_num, r, c] = cls_batch[b][i]
                loc_t4[b, st_box_idx : ed_box_idx, r, c] = (gt_box_batch_idx - df_box_batch_idx) / common_params.loc_var
            elif fmap_layer == 4:
                cls_t5[b, df_box_num, r, c] = cls_batch[b][i]
                loc_t5[b, st_box_idx : ed_box_idx, r, c] = (gt_box_batch_idx - df_box_batch_idx) / common_params.loc_var
            elif fmap_layer == 5:
                cls_t6[b, df_box_num, r, c] = cls_batch[b][i]
                loc_t6[b, st_box_idx : ed_box_idx, r, c] = (gt_box_batch_idx - df_box_batch_idx) / common_params.loc_var

    # 1〜6階層目の教示confidence mapをVariableにする
    with chainer.using_config('enable_backprop', False):
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

    # 1〜6階層目の教示localization mapをVariableにする
    with chainer.using_config('enable_backprop', False):
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
    #loss_cls = loss_cls * 0.4

    # localization mapのloss
    # オリジナル実装ではSmooth L1 Lossだが、本ソースコードではMean Squared Errorを使用
    # Smooth L1 Lossで誤差を求める場合は「F.huber_loss(x, t, delta = 1.0)」
    loss_loc = F.mean_squared_error(Loc, Loc_T)
    #loss_loc = loss_cls * 0.5
    #loss_loc = F.mean_absolute_error(Loc, LT)
    #loss_loc = sum_squared_error(Loc, LT)

    #segmentationのloss
    loss_seg = F.softmax_cross_entropy(Seg, seg_label)

    # nanのチェック
    if math.isnan(loss_cls.data):
        print('Warn: loss_cls.data is nan')
    if math.isnan(loss_loc.data):
        print('Warn: loss_loc.data is nan')
    if math.isnan(loss_seg.data):
        print('Warn: loss_seg.data is nan')

    # lossをstdout
    print('confidence loss   : ' + str(loss_cls.data))
    print('localization loss : ' + str(loss_loc.data))
    print('segmentation loss : ' + str(loss_seg.data))

    # lossをfileout
    file = open(today_dir_path + '/SSD_seg_loss/loss_cls.txt', 'a')
    out_line = '{} \n'.format(loss_cls.data)
    file.write(out_line)
    file.close()

    file = open(today_dir_path + '/SSD_seg_loss/loss_loc.txt', 'a')
    out_line = '{} \n'.format(loss_loc.data)
    file.write(out_line)
    file.close()

    file = open(today_dir_path + '/SSD_seg_loss/loss_seg.txt', 'a')
    out_line = '{} \n'.format(loss_seg.data)
    file.write(out_line)
    file.close()

    del cls_t1, cls_t2, cls_t3, cls_t4, cls_t5, cls_t6
    del loc_t1, loc_t2, loc_t3, loc_t4, loc_t5, loc_t6
    del cls_t1_data, cls_t2_data, cls_t3_data, cls_t4_data, cls_t5_data, cls_t6_data
    del loc_t1_data, loc_t2_data, loc_t3_data, loc_t4_data, loc_t5_data, loc_t6_data
    print(loss_cls, loss_loc, loss_seg)
    print(type(loss_cls), type(loss_loc), type(loss_seg))
    return loss_cls, loss_loc, loss_seg


def main():
    save_dir_suffix = "Chainer5_Trainer"

    #VGGNetの読み込み
    print('VGG Netの読み込み中...')
    vgg_model = VGGNet()
    serializers.load_npz('./pretrained_model/VGGNet_for_SSD.model', vgg_model) #VGGモデルの読み込み
    print('-> 読み込み完了')

    #SSDNetの読み込み
    ssd_model = SSDNet()

    #GPUを使う
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        vgg_model.to_gpu()
        ssd_model.to_gpu()

    copy_model(vgg_model, ssd_model)

    del vgg_model

    # Output dir
    today = datetime.datetime.today()   #日付を取得
    today_dir = str(today.year) + '-' + str('%02d' % today.month) + '-' + str('%02d' % today.day) + '@' + str('%02d' % today.hour) + '-' + str('%02d' % today.minute) + '-' + str('%02d' % today.second)
    today_dir_path = path.join(common_params.save_model_dir, today_dir + save_dir_suffix)

    if not path.exists(today_dir_path):
        os.mkdir(today_dir_path)
        os.mkdir(today_dir_path + '/SSD_seg_loss/')
    loss_log_path = today_dir_path + '/loss.txt'

    output_dir = today_dir_path

    # Dataset import
    train = Dataset("./augimg_name_list.txt", "./seglabel_name_list.txt", True)
    #train.get_example(0)
    train_chain = TrainChain(ssd_model, today_dir_path)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    # Setup optimizer
    optimizer = optimizers.MomentumSGD(lr = common_params.learning_rate, momentum = common_params.momentum)
    optimizer.setup(train_chain)
    optimizer.add_hook(chainer.optimizer.WeightDecay(common_params.weight_decay))
    """
    optimizer = optimizers.MomentumSGD()
    optimizer.setup(train_chain)
    for param in train_chain.params():
        if param.name == 'b':
            param.update_rule.add_hook(GradientScaling(2))
        else:
            param.update_rule.add_hook(chainer.optimizer.WeightDecay(0.0005))
    """
    #optimizer.add_hook(DelGradient(["Loc1", "Cls1", "esconv1", "esconv2", "esconv3", "esconv4", "esconv5"]))
    #optimizer.add_hook(DelGradient(["Loc1", "Cls1", "Loc2", "Cls2", "Loc3", "Cls3", "Loc4", "Cls4", "Loc5", "Cls5", "Loc6", "Cls6"]))

    # Trainer/Updater setup
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), output_dir)
    lr_reduce_epoch = range(common_params.lr_change_epoch, args.epoch)
    trainer.extend(
        extensions.ExponentialShift('lr', rate=common_params.lr_step, init=common_params.learning_rate, target=common_params.lower_lr),
        trigger=triggers.ManualScheduleTrigger(lr_reduce_epoch, 'epoch'))
    trainer.extend(set_epoch_num(ssd_model, updater.epoch))

    # Train logging setup
    log_interval = 10, 'iteration'
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'lr',
         'main/loss', 'main/loss/cls', 'main/loss/loc', 'main/loss/seg']),
        trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # Model saving setup
    trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))
    trainer.extend(
        extensions.snapshot_object(ssd_model, 'SSD_Seg_epoch_{.updater.epoch}.model'),
        trigger=(1, 'epoch'))

    if args.resume:
        serializers.load_npz(args.resume, trainer)

    trainer.run()
    print("\nExit Training\n")

if __name__ == '__main__':
    main()
