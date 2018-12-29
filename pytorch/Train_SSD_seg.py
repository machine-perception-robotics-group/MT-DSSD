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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchnet as tnt
from torchnet.engine import Engine

import common_params
from data_augmentation_seg import trainAugmentation

from SSD_seg_Net import SSDNet
from VGG_Net import VGGNet

parser = argparse.ArgumentParser()  #パーサーを作る
parser.add_argument('--batchsize', '-B', type = int, default = 5, help = 'Learning minibatch size') #引数の追加
parser.add_argument('--epoch', '-E', default = 150, type = int, help='Number of epochs to learn')    #引数の追加
parser.add_argument('--gpu', '-g', default = 0, type = int, help='GPU ID (negative value indicates CPU)')  #引数の追加
parser.add_argument('--loaderjob', '-j', default = 4, type=int, help='Number of parallel data loading processes')   #引数の追加
parser.add_argument('--segmentignore', '-s', default = 0, type=int, help='Even if segmentation image is not found, continue learning (1:True/0:False)')   #引数の追加
parser.add_argument('--resume')
args = parser.parse_args()  #引数を解析

# epoch数
n_epoch = args.epoch

# バッチサイズ
batchsize = args.batchsize

#GPUを使う
if args.gpu >= 0:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

TIMER = False
DEBUG = False

def copy_model_npz(src, dst):   #src:vgg_model(npz) dst:ssd_model
    for src_layer_name in src.keys():
        src_layer_name_full = src_layer_name.replace("/b", ".bias").replace("/W", ".weight")
        found = False
        for dst_param in dst.named_parameters():
            if src_layer_name_full == str(dst_param[0]):
                # name match
                found = True
                if src[src_layer_name].shape == dst_param[1].shape:
                    # param match
                    #print(src_param[1].shape)
                    #print(dst_param[1].data)
                    #print(src_param[1].data)
                    dst_param[1].data = torch.tensor(src[src_layer_name])
                    print("Copy {}".format(src_layer_name_full))
                else:
                    #param mismatch
                    print("Ignore {} because of parameter mismatch. src: {}, dst: {}".format(src_layer_name_full, src[src_layer_name].shape, dst_param[1].shape))
                break
        if not found: print("Not found {} in src model".format(src_layer_name_full))


def copy_model(src, dst):   #src:vgg_model dst:ssd_model
    assert isinstance(src, nn.Module)
    assert isinstance(dst, nn.Module)
    for src_param in src.named_parameters():
        found = False
        for dst_param in dst.named_parameters():
            if str(src_param[0]) == str(dst_param[0]):
                # name match
                found = True
                if src_param[1].shape == dst_param[1].shape:
                    # param match
                    #print(src_param[1].shape)
                    #print(dst_param[1].data)
                    #print(src_param[1].data)
                    dst_param[1].data = src_param[1].data
                    print("Copy {}".format(src_param[0]))
                else:
                    #param mismatch
                    print("Ignore {} because of parameter mismatch".format(src_param[0]))
                break
        if not found: print("Not found {} in src model".format(src_param[0]))


class MTDataset(Dataset):
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

    def __getitem__(self, i):
        device_ = "cpu"
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
                seg_labelimg = torch.ones((common_params.insize, common_params.insize)) * -1
            else:
                print("[ERROR] Cannot read segmentation label")
                print(common_params.images_dir + "/train/seglabel/" + seg_label_path + ".png")
                print("引数 -segmentignore 1を与えれば無視してB-Boxのみで学習します(Segmentation誤差を伝播させません)")
                sys.exit(1)

        # 画像をSSDの入力サイズにリサイズ
        input_img = cv.resize(color_img, (common_params.insize, common_params.insize), interpolation = cv.INTER_CUBIC)  #バイキュビック補間

        input_seglabel = cv.resize(seg_labelimg, (common_params.insize, common_params.insize), interpolation = cv.INTER_NEAREST)

        if augmentation == 1:
            input_img, input_seglabel = trainAugmentation(input_img, border_pixels, crop_param, hsv_param, flip_type, input_seglabel)   #data augmentation

        if self.confing_image:
            conf_img = input_img.copy()

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

        # segmentationのignore class(255)を-1にする
        input_seglabel = input_seglabel.astype(np.int64) #uintからintにしないと負の値が入らない
        input_seglabel[input_seglabel==255] = -1

        # padding ???(random) -> 8732(dfbox max size)
        max_boxes = 8732 * 4

        if len(gt_boxes) == 0:
            gt_boxes = np.ones((max_boxes, 4)) * -100
        elif len(gt_boxes) != max_boxes:
            gt_boxes = np.pad(np.array(gt_boxes), [(0,max_boxes-len(gt_boxes)), (0,0)], 'constant', constant_values=-100)

        if len(df_boxes) == 0:
            df_boxes = np.ones((max_boxes, 4)) * -100
        elif len(df_boxes) != max_boxes:
            df_boxes = np.pad(np.array(df_boxes), [(0,max_boxes-len(df_boxes)), (0,0)], 'constant', constant_values=-100)

        if len(indices) == 0:
            indices = np.ones((max_boxes, 4)) * -100
        elif len(indices) != max_boxes:
            indices = np.pad(np.array(indices), [(0,max_boxes-len(indices)), (0,0)], 'constant', constant_values=-100)

        if len(classes) == 0:
            classes = np.ones((max_boxes,)) * -100
        elif len(classes) != max_boxes:
            classes = np.pad(np.array(classes), (0,max_boxes-len(classes)), 'constant', constant_values=-100)

        if DEBUG:
            print("input_img", np.array(input_img).shape, type(input_img[0][0][0]))
            print("gt_boxes", np.array(gt_boxes).shape, type(gt_boxes[0][0]))
            print("df_boxes", np.array(df_boxes).shape, type(df_boxes[0][0]))
            print("indices", np.array(indices).shape, type(indices[0][0]))
            print("classes", np.array(classes).shape, type(classes[0]))
            print("conf_img", np.array(conf_img).shape, type(conf_img[0][0][0]))
            print("input_seglabel", np.array(input_seglabel).shape, type(input_seglabel[0][0]))
            print("-----------------")
        """
        if len(gt_boxes) == 0:
            gt_boxes = np.zeros((max_boxes, 4))
        elif len(gt_boxes) != max_boxes:
            gt_boxes = np.pad(np.array(gt_boxes), [(0,max_boxes-len(gt_boxes)), (0,0)], 'constant', constant_values=0)

        if len(df_boxes) == 0:
            df_boxes = np.zeros((max_boxes, 4))
        elif len(df_boxes) != max_boxes:
            df_boxes = np.pad(np.array(df_boxes), [(0,max_boxes-len(df_boxes)), (0,0)], 'constant', constant_values=0)

        if len(indices) == 0:
            indices = np.zeros((max_boxes, 4))
        elif len(indices) != max_boxes:
            indices = np.pad(np.array(indices), [(0,max_boxes-len(indices)), (0,0)], 'constant', constant_values=0)

        if len(classes) == 0:
            classes = np.zeros((max_boxes,))
        elif len(classes) != max_boxes:
            classes = np.pad(np.array(classes), (0,max_boxes-len(classes)), 'constant', constant_values=0)
        """

        # list to numpy
        input_img = torch.tensor(input_img, device=device_)
        gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32, device=device_)
        df_boxes = torch.tensor(df_boxes, dtype=torch.float32, device=device_)
        indices = torch.tensor(indices, dtype=torch.int64, device=device_)
        classes = torch.tensor(classes, dtype=torch.int64, device=device_)
        conf_img = torch.tensor(conf_img, device=device_)
        input_seglabel = torch.tensor(input_seglabel, device=device_)

        #print("final:" + str(np.max(input_seglabel))) #デバッグ用 seg教師信号の値確認 255以外で一番でかいやつを出力

        return input_img, gt_boxes, df_boxes, indices, classes, conf_img, input_seglabel


class MTLoss(nn.Module):
    def __init__(self, today_dir_path):
        super(MTLoss, self).__init__()
        self.today_dir_path = today_dir_path

    def forward(self, Loc, Cls, Seg, gt_box_batch, df_box_batch, idx_batch, cls_batch, bat_s, mining, seg_label):
        if TIMER: start = time.time()
        Loc.to(device)
        Cls.to(device)
        Seg.to(device)
        gt_box_batch.to(device)
        df_box_batch.to(device)
        idx_batch.to(device)
        cls_batch.to(device)
        seg_label.to(device)
        if DEBUG: print("Loc", type(Loc), Loc.is_cuda)
        device_ = "cpu"
        if TIMER: print("Loss_init:", time.time()-start)

        today_dir_path = self.today_dir_path

        if TIMER: start = time.time()
        if mining:
            # hard negative mining有効時のクラスラベル
            #cls_t = torch.ones((6, bat_s, common_params.num_boxes[0], common_params.map_sizes[0], common_params.map_sizes[0]), dtype = torch.int64, device = device_) * -1
            cls_t1 = torch.ones((bat_s, common_params.num_boxes[0], common_params.map_sizes[0], common_params.map_sizes[0]), dtype = torch.int64, device = device_) * -1
            cls_t2 = torch.ones((bat_s, common_params.num_boxes[1], common_params.map_sizes[1], common_params.map_sizes[1]), dtype = torch.int64, device = device_) * -1
            cls_t3 = torch.ones((bat_s, common_params.num_boxes[2], common_params.map_sizes[2], common_params.map_sizes[2]), dtype = torch.int64, device = device_) * -1
            cls_t4 = torch.ones((bat_s, common_params.num_boxes[3], common_params.map_sizes[3], common_params.map_sizes[3]), dtype = torch.int64, device = device_) * -1
            cls_t5 = torch.ones((bat_s, common_params.num_boxes[4], common_params.map_sizes[4], common_params.map_sizes[4]), dtype = torch.int64, device = device_) * -1
            cls_t6 = torch.ones((bat_s, common_params.num_boxes[5], common_params.map_sizes[5], common_params.map_sizes[5]), dtype = torch.int64, device = device_) * -1
        else:
            # hard negative mining無効時のクラスラベル
            #cls_t = torch.zeros((6, bat_s, common_params.num_boxes[0], common_params.map_sizes[0], common_params.map_sizes[0]), dtype = torch.int64, device = device_)
            cls_t1 = torch.zeros((bat_s, common_params.num_boxes[0], common_params.map_sizes[0], common_params.map_sizes[0]), dtype = torch.int64, device = device_)
            cls_t2 = torch.zeros((bat_s, common_params.num_boxes[1], common_params.map_sizes[1], common_params.map_sizes[1]), dtype = torch.int64, device = device_)
            cls_t3 = torch.zeros((bat_s, common_params.num_boxes[2], common_params.map_sizes[2], common_params.map_sizes[2]), dtype = torch.int64, device = device_)
            cls_t4 = torch.zeros((bat_s, common_params.num_boxes[3], common_params.map_sizes[3], common_params.map_sizes[3]), dtype = torch.int64, device = device_)
            cls_t5 = torch.zeros((bat_s, common_params.num_boxes[4], common_params.map_sizes[4], common_params.map_sizes[4]), dtype = torch.int64, device = device_)
            cls_t6 = torch.zeros((bat_s, common_params.num_boxes[5], common_params.map_sizes[5], common_params.map_sizes[5]), dtype = torch.int64, device = device_)

        # bounding boxのオフセットベクトルの教示データ
        #loc_t = torch.zeros((6, bat_s, common_params.num_boxes[0] * common_params.num_of_offset_dims, common_params.map_sizes[0], common_params.map_sizes[0]), dtype = torch.float32, device = device_)
        loc_t1 = torch.zeros((bat_s, common_params.num_boxes[0] * common_params.num_of_offset_dims, common_params.map_sizes[0], common_params.map_sizes[0]), dtype = torch.float32, device = device_)
        loc_t2 = torch.zeros((bat_s, common_params.num_boxes[1] * common_params.num_of_offset_dims, common_params.map_sizes[1], common_params.map_sizes[1]), dtype = torch.float32, device = device_)
        loc_t3 = torch.zeros((bat_s, common_params.num_boxes[2] * common_params.num_of_offset_dims, common_params.map_sizes[2], common_params.map_sizes[2]), dtype = torch.float32, device = device_)
        loc_t4 = torch.zeros((bat_s, common_params.num_boxes[3] * common_params.num_of_offset_dims, common_params.map_sizes[3], common_params.map_sizes[3]), dtype = torch.float32, device = device_)
        loc_t5 = torch.zeros((bat_s, common_params.num_boxes[4] * common_params.num_of_offset_dims, common_params.map_sizes[4], common_params.map_sizes[4]), dtype = torch.float32, device = device_)
        loc_t6 = torch.zeros((bat_s, common_params.num_boxes[5] * common_params.num_of_offset_dims, common_params.map_sizes[5], common_params.map_sizes[5]), dtype = torch.float32, device = device_)
        if TIMER: print("Loss_init_tensor:", time.time()-start)
        cls_t = ([cls_t1, cls_t2, cls_t3, cls_t4, cls_t5, cls_t5])
        loc_t = ([loc_t1, loc_t2, loc_t3, loc_t4, loc_t5, loc_t6])
        if DEBUG: print("cls_t1", type(cls_t1), cls_t1.is_cuda)

        if TIMER: start = time.time()
        if DEBUG: print("idx_batch", len(idx_batch))
        if DEBUG: print("idx_batch[b]", len(idx_batch[0]))
        if DEBUG: print("idx_batch", np.array(idx_batch).shape, np.array(idx_batch))
        if DEBUG: print("cls_batch", np.array(cls_batch).shape, np.array(cls_batch))
        for b in xrange(0, len(idx_batch)):
            for i in xrange(0, len(idx_batch[b])):
                if int(idx_batch[b][i][1]) == -100: break

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

                cls_t[fmap_layer][b, df_box_num, r, c] = cls_batch[b][i]
                loc_t[fmap_layer][b, st_box_idx : ed_box_idx, r, c] = (gt_box_batch_idx - df_box_batch_idx) / common_params.loc_var
        if DEBUG:
            print("")
            for j in range(0, 6):
                print("cls_t[" + str(j) + "]", torch.min(cls_t[j]))
                print("loc_t[" + str(j) + "]", torch.min(loc_t[j]))

        if DEBUG: print("cls_t", np.array(cls_t[0]).shape, np.array(cls_t[0]))
        if DEBUG: print("loc_t", np.array(loc_t[0]).shape, np.array(loc_t[0]))
        if TIMER: print("Loss_forloop:", time.time()-start)

        if DEBUG: print("cls_t1", type(cls_t1), cls_t1.is_cuda)
        cls_t1_data = cls_t1
        cls_t2_data = cls_t2
        cls_t3_data = cls_t3
        cls_t4_data = cls_t4
        cls_t5_data = cls_t5
        cls_t6_data = cls_t6
        if DEBUG: print("cls_t1_data", type(cls_t1_data), cls_t1_data.is_cuda)

        # 1〜6階層目の教示confidence mapの次元を(バッチ数, DF box数, 高さ, 幅)から(バッチ数, 高さ, 幅, DF box数)に転置
        cls_t1_data = cls_t1_data.permute(0, 2, 3, 1)
        cls_t2_data = cls_t2_data.permute(0, 2, 3, 1)
        cls_t3_data = cls_t3_data.permute(0, 2, 3, 1)
        cls_t4_data = cls_t4_data.permute(0, 2, 3, 1)
        cls_t5_data = cls_t5_data.permute(0, 2, 3, 1)
        cls_t6_data = cls_t6_data.permute(0, 2, 3, 1)

        # 1〜6階層目の教示confidence mapの各次元数を(バッチ数, 高さ, 幅, DF box数)から(バッチ数 * 高さ * 幅 * DF box数)にreshape
        cls_t1_data = cls_t1_data.contiguous()
        cls_t2_data = cls_t2_data.contiguous()
        cls_t3_data = cls_t3_data.contiguous()
        cls_t4_data = cls_t4_data.contiguous()
        cls_t5_data = cls_t5_data.contiguous()
        cls_t6_data = cls_t6_data.contiguous()
        cls_t1_data = cls_t1_data.view(cls_t1_data.data.shape[0] * cls_t1_data.data.shape[1] * cls_t1_data.data.shape[2] * common_params.num_boxes[0])
        cls_t2_data = cls_t2_data.view(cls_t2_data.data.shape[0] * cls_t2_data.data.shape[1] * cls_t2_data.data.shape[2] * common_params.num_boxes[1])
        cls_t3_data = cls_t3_data.view(cls_t3_data.data.shape[0] * cls_t3_data.data.shape[1] * cls_t3_data.data.shape[2] * common_params.num_boxes[2])
        cls_t4_data = cls_t4_data.view(cls_t4_data.data.shape[0] * cls_t4_data.data.shape[1] * cls_t4_data.data.shape[2] * common_params.num_boxes[3])
        cls_t5_data = cls_t5_data.view(cls_t5_data.data.shape[0] * cls_t5_data.data.shape[1] * cls_t5_data.data.shape[2] * common_params.num_boxes[4])
        cls_t6_data = cls_t6_data.view(cls_t6_data.data.shape[0] * cls_t6_data.data.shape[1] * cls_t6_data.data.shape[2] * common_params.num_boxes[5])

        loc_t1_data = loc_t1
        loc_t2_data = loc_t2
        loc_t3_data = loc_t3
        loc_t4_data = loc_t4
        loc_t5_data = loc_t5
        loc_t6_data = loc_t6

        # 1〜6階層目の教示localization mapの次元を(バッチ数, オフセット次元数 * DF box数, 高さ, 幅)から(バッチ数, 高さ, 幅, オフセット次元数 * DF box数)に転置
        loc_t1_data = loc_t1_data.permute(0, 2, 3, 1)
        loc_t2_data = loc_t2_data.permute(0, 2, 3, 1)
        loc_t3_data = loc_t3_data.permute(0, 2, 3, 1)
        loc_t4_data = loc_t4_data.permute(0, 2, 3, 1)
        loc_t5_data = loc_t5_data.permute(0, 2, 3, 1)
        loc_t6_data = loc_t6_data.permute(0, 2, 3, 1)

        # 1〜6階層目の教示localization mapの各次元数を(バッチ数, 高さ, 幅, オフセット次元数 * DF box数)から(バッチ数 * 高さ * 幅 * DF box数, オフセット次元数)にreshape
        loc_t1_data = loc_t1_data.contiguous()
        loc_t2_data = loc_t2_data.contiguous()
        loc_t3_data = loc_t3_data.contiguous()
        loc_t4_data = loc_t4_data.contiguous()
        loc_t5_data = loc_t5_data.contiguous()
        loc_t6_data = loc_t6_data.contiguous()
        loc_t1_data = loc_t1_data.view(loc_t1_data.data.shape[0] * loc_t1_data.data.shape[1] * loc_t1_data.data.shape[2] * common_params.num_boxes[0], int(loc_t1_data.data.shape[3] / common_params.num_boxes[0]))
        loc_t2_data = loc_t2_data.view(loc_t2_data.data.shape[0] * loc_t2_data.data.shape[1] * loc_t2_data.data.shape[2] * common_params.num_boxes[1], int(loc_t2_data.data.shape[3] / common_params.num_boxes[1]))
        loc_t3_data = loc_t3_data.view(loc_t3_data.data.shape[0] * loc_t3_data.data.shape[1] * loc_t3_data.data.shape[2] * common_params.num_boxes[2], int(loc_t3_data.data.shape[3] / common_params.num_boxes[2]))
        loc_t4_data = loc_t4_data.view(loc_t4_data.data.shape[0] * loc_t4_data.data.shape[1] * loc_t4_data.data.shape[2] * common_params.num_boxes[3], int(loc_t4_data.data.shape[3] / common_params.num_boxes[3]))
        loc_t5_data = loc_t5_data.view(loc_t5_data.data.shape[0] * loc_t5_data.data.shape[1] * loc_t5_data.data.shape[2] * common_params.num_boxes[4], int(loc_t5_data.data.shape[3] / common_params.num_boxes[4]))
        loc_t6_data = loc_t6_data.view(loc_t6_data.data.shape[0] * loc_t6_data.data.shape[1] * loc_t6_data.data.shape[2] * common_params.num_boxes[5], int(loc_t6_data.data.shape[3] / common_params.num_boxes[5]))

        # 1〜6階層目の教示confidence mapを結合
        Cls_T = torch.cat([cls_t1_data, cls_t2_data, cls_t3_data, cls_t4_data, cls_t5_data, cls_t6_data], dim = 0)

        # 1〜6階層目の教示localization mapを結合
        Loc_T = torch.cat([loc_t1_data, loc_t2_data, loc_t3_data, loc_t4_data, loc_t5_data, loc_t6_data], dim = 0)
        if DEBUG: print(type(Cls_T))
        if DEBUG: print(type(Loc_T))
        if DEBUG: print("Cls_T", type(Cls_T), Cls_T.is_cuda)
        if DEBUG: print("Loc_T", type(Loc_T), Loc_T.is_cuda)
        # loss計算
        x_entropy = nn.CrossEntropyLoss(ignore_index=-1)
        MSE = nn.MSELoss()

        # confidence mapのloss
        loss_cls = x_entropy(Cls.to(device), Cls_T.to(device))
        #loss_cls = loss_cls * 0.4

        # localization mapのloss
        # オリジナル実装ではSmooth L1 Lossだが、本ソースコードではMean Squared Errorを使用
        loss_loc = MSE(Loc.to(device), Loc_T.to(device))

        #segmentationのloss
        loss_seg = x_entropy(Seg.to(device), seg_label.to(device))

        # nanのチェック
        if math.isnan(loss_cls.data):
            print("[Error] loss_cls.data is nan")
            exit(1)
        if math.isnan(loss_loc.data):
            print("[Error] loss_loc.data is nan")
            exit(1)
        if math.isnan(loss_seg.data):
            print("[Error] loss_seg.data is nan")
            exit(1)

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

        return loss_cls, loss_loc, loss_seg


def main():
    # 事前学習VGGNetの読み込み
    """
    print('VGG Netの読み込み中...')
    vgg_model = VGGNet()
    vgg_b_w = torch.load("./pretrained_model/VGG_ILSVRC_16_layers_fc_reduced.caffemodel.pth")
    del vgg_b_w['fc8.bias'], vgg_b_w['fc8.weight']
    vgg_model.load_state_dict(vgg_b_w)
    print('-> 読み込み完了')
    """
    # SSDNetの読み込み
    ssd_model = SSDNet()


    # 事前学習モデルをコピー
    #copy_model(vgg_model, ssd_model)
    copy_model_npz(np.load("../pretrained_model/VGGNet_for_SSD.model"), ssd_model)
    #del vgg_model

    if args.gpu >= 0:
        #vgg_model.to(device)
        ssd_model.to(device)

    # 学習再開
    """
    if args.resume:
        ssd_model.load_state_dict(torch.load(args.resume))
    """

    # Output dir
    today = datetime.datetime.today()   #日付を取得
    today_dir = str(today.year) + '-' + str('%02d' % today.month) + '-' + str('%02d' % today.day) + '@' + str('%02d' % today.hour) + '-' + str('%02d' % today.minute) + '-' + str('%02d' % today.second)
    save_dir_suffix = "_PyTorch1"
    today_dir_path = path.join(common_params.save_model_dir, today_dir + save_dir_suffix)

    if not path.exists(today_dir_path):
        os.mkdir(today_dir_path)
        os.mkdir(path.join(today_dir_path, "SSD_seg_loss"))
    loss_fout = open(path.join(today_dir_path, "loss.txt"), 'w')
    save_model_path = path.join(today_dir_path, "model")
    save_optimizer_path = path.join(today_dir_path, "optimizer")
    if not path.exists(save_model_path):
        os.mkdir(save_model_path)
    if not path.exists(save_optimizer_path):
        os.mkdir(save_optimizer_path)


    # Dataset import
    train_dataset = MTDataset("./augimg_name_list.txt", "./seglabel_name_list.txt", True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=args.loaderjob)
    # Setup optimizer
    optimizer = optim.SGD(ssd_model.parameters(), lr = common_params.learning_rate, momentum = common_params.momentum, weight_decay = common_params.weight_decay)

    # loss function
    loss_function = MTLoss(today_dir_path)

    all_iter = 0
    begin_at = time.time()
    for num_epoch in range(args.epoch):
        running_loss_cls = 0.0
        running_loss_loc = 0.0
        running_loss_seg = 0.0
        running_loss = 0.0

        for g in optimizer.param_groups:
            # Change learning rate
            if (num_epoch + 1) >= common_params.lr_change_epoch:
                g['lr'] *= common_params.lr_step if g['lr'] >= common_params.lower_lr else common_params.lower_lr
            now_lr = g['lr']

        print ("[Epoch: {} start] Learning rate: {} ({})\n".format(num_epoch + 1, now_lr, today_dir))

        for num_batch, data in enumerate(train_dataloader):
            all_iter += 1
            if TIMER: start = time.time()
            img_batch, gt_box_batch, df_box_batch, idx_batch, cls_batch, conf_img_batch, seglabel_batch = data
            if TIMER: print("Dataget:", time.time()-start)

            if TIMER: start = time.time()
            train_img = img_batch.to(device)
            seg_label = seglabel_batch
            if TIMER: print("InitData:", time.time()-start)

            # SSD net forward
            if TIMER: start = time.time()
            Loc1, Cls1, Loc2, Cls2, Loc3, Cls3, Loc4, Cls4, Loc5, Cls5, Loc6, Cls6, Seg = ssd_model(train_img, train=True)
            if TIMER: print("Forward:", time.time()-start)

            if TIMER: start = time.time()
            # ネットワークから出力されたconfidence mapを1〜6階層目まで結合
            Loc = torch.cat([Loc1, Loc2, Loc3, Loc4, Loc5, Loc6], dim = 0)

            # ネットワークから出力されたlocalization mapを1〜6階層目まで結合
            Cls = torch.cat([Cls1, Cls2, Cls3, Cls4, Cls5, Cls6], dim = 0)

            if TIMER: print("ConvertTensor:", time.time()-start)

            # lossを計算
            if TIMER: start = time.time()

            # epochが4回に1回の割合でhard negative mining有効
            if num_epoch+1 % 4 == 0:
                mining = True
            else:
                mining = False

            loss_cls, loss_loc, loss_seg = loss_function(Loc, Cls, Seg, gt_box_batch, df_box_batch, idx_batch, cls_batch, batchsize, mining, seg_label)
            if TIMER: print("Loss:", time.time()-start)

            loss = (loss_cls + loss_loc + loss_seg) / 3.
            if TIMER: start = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if TIMER: print("Backward:", time.time()-start)

            sys.stdout.write("\rEpoch: {}, Iter: {}, Time: {}, Loss:{} (cls: {}, loc: {}, seg: {})".format(num_epoch+1, all_iter, datetime.timedelta(seconds = time.time() - begin_at), loss.data, loss_cls.data, loss_loc.data, loss_seg.data))
            sys.stdout.flush()
            time.sleep(0.01)

            running_loss_cls += loss_cls
            running_loss_loc += loss_loc
            running_loss_seg += loss_seg
            running_loss += loss
            if num_batch+1 % 100 == 0:
                print("\n[Epoch {} Iter {}] Total Iter: {}, Time: {}, Ave_Loss:{} (cls: {}, loc: {}, seg: {}\n) ".format(num_epoch+1, num_batch, all_iter, datetime.timedelta(seconds = time.time() - begin_at), running_loss/100., running_loss_cls/100., running_loss_loc/100., running_loss_seg/100.))
                out_line = "{} {} {} {} {} {} {} {} {}\n".format(num_epoch+1, now_lr, all_iter, datetime.timedelta(seconds = time.time() - begin_at), running_loss/100., running_loss_cls/100., running_loss_loc/100., running_loss_seg/100.)
                loss_fout.write(out_line)
                running_loss_cls = 0.0
                running_loss_loc = 0.0
                running_loss_seg = 0.0
                running_loss = 0.0

        print("\n[Epoch {} done] Save model.".format(num_epoch+1))
        if num_epoch+1 % 4 == 0:
            suffix = "_with_mining"
        else:
            suffix = "_without_mining"

        torch.save(ssd_model.state_dict(), save_model_path + "/SSD_Seg_epoch_" + str(num_epoch + 1) + suffix + ".pth")
        torch.save(optimizer.state_dict(), save_optimizer_path + "/SSD_Seg_epoch_" + str(num_epoch + 1) + suffix + "_optim.pth")


    """
    def h(sample):
        inputs = Variable(sample[0].float() / 255.0)
        targets = Variable(torch.LongTensor(sample[1]))
        o = f(params, inputs, sample[2])
        return F.cross_entropy(o, targets), o
    """

    """
    # TNT setup
    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    #classerr = tnt.meter.ClassErrorMeter(accuracy=True)

    def reset_meters():
        #classerr.reset()
        meter_loss.reset()

    def on_sample(state):
        state['sample'].append(state['train'])

    def on_forward(state):
        #classerr.add(state['output'].data,
                     torch.LongTensor(state['sample'][1]))
        meter_loss.add(state['loss'].data[0])

    def on_start_epoch(state):
        reset_meters()
        state['iterator'] = tqdm(state['iterator'])

    def on_end_epoch(state):
        #print('Training loss: %.4f, accuracy: %.2f%%' % (meter_loss.value()[0], classerr.value()[0]))
        print('Training loss: %.4f' % (meter_loss.value()[0]))
        reset_meters()

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.train(ssd_model, iterator, maxepoch=150, optimizer=optimizer)
    """

    print("\nExit Training\n")
    loss_fout.close()

if __name__ == '__main__':
    main()
