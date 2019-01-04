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
from tqdm import tqdm

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
#import torchnet as tnt
#from torchnet.engine import Engine

import common_params
from data_augmentation_seg import trainAugmentation

from SSD_seg_Net import SSDNet
from VGG_Net import VGGNet

parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', '-B', type = int, default = 5, help = "Learning minibatch size")
parser.add_argument('--epoch', '-E', default = 150, type = int, help="Number of epochs to learn")
parser.add_argument('--gpu', '-g', default = 0, type = int, help="GPU ID (negative value indicates CPU, 100 indicates all GPU)")
parser.add_argument('--loaderjob', '-j', default = 4, type=int, help="Number of parallel data loading processes")
parser.add_argument('--segmentignore', '-s', default = 0, type=int, help="Even if segmentation image is not found, continue learning (1:True / 0:False)")
parser.add_argument('--pretrain_type', '-p', default = 0, type=int, help="Pre-train type (1:load pytorch model / 2:load chainer npz / 3:load VGG from torchvision / other: Don't use pretrained model) ")
parser.add_argument('--resume', help="Learning model path (if resume learning)")
args = parser.parse_args()

#GPUを使う
if args.gpu >= 100:
    device = torch.device("cuda")
elif args.gpu >= 0:
    device = torch.device("cuda:"+str(args.gpu))
else:
    device = torch.device("cpu")

TIMER = False

def copy_model_npz(src, dst):   #src:vgg_model(npz) dst:ssd_model
    for src_layer_name in src.keys():
        #src_layer_name_full = src_layer_name.replace("/b", ".bias").replace("/W", ".weight")
        src_layer_name_full = src_layer_name.replace("/W", ".weight")
        if src_layer_name.find("/b") != -1: continue
        found = False
        for dst_param in dst.named_parameters():
            if src_layer_name_full == str(dst_param[0]):
                # name match
                found = True
                if src[src_layer_name].shape == dst_param[1].shape:
                    # param match
                    dst_param[1].data = torch.tensor(src[src_layer_name])
                    print("Copy {}".format(src_layer_name_full))
                else:
                    #param mismatch
                    print("Ignore {} because of parameter mismatch. src: {}, dst: {}".format(src_layer_name_full, src[src_layer_name].shape, dst_param[1].shape))
                break
        if not found: print("Not found {} in dst model".format(src_layer_name_full))


def copy_model(src, dst):   #src:vgg_model dst:ssd_model
    assert isinstance(src, nn.Module)
    assert isinstance(dst, nn.Module)
    for src_param in src.named_parameters():
        found = False
        if str(src_param[0]).find("bias") != -1: continue
        for dst_param in dst.named_parameters():
            if str(src_param[0]) == str(dst_param[0]):
                # name match
                found = True
                if src_param[1].shape == dst_param[1].shape:
                    # param match
                    dst_param[1].data = src_param[1].data
                    print("Copy {}".format(src_param[0]))
                else:
                    #param mismatch
                    print("Ignore {} because of parameter mismatch".format(src_param[0]))
                break
        if not found: print("Not found {} in dst model".format(src_param[0]))

def copy_model_torchvision(src, dst, src_tbl, dst_tbl):   #src:vgg_model dst:ssd_model
    assert isinstance(src, nn.Module)
    assert isinstance(dst, nn.Module)
    for src_param in src.named_parameters():
        found = False
        if str(src_param[0]).find("bias") != -1: continue
        for dst_param in dst.named_parameters():
            if src_tbl.count(str(src_param[0])) != 0:
                src_name = dst_tbl[src_tbl.index(str(src_param[0]))]
            else:
                break
            if src_name == str(dst_param[0]):
                # name match
                found = True
                if src_param[1].shape == dst_param[1].shape:
                    # param match
                    dst_param[1].data = src_param[1].data
                    print("Copy {} ({})".format(src_param[0], str(dst_param[0])))
                else:
                    #param mismatch
                    print("Ignore {} because of parameter mismatch".format(src_param[0]))
                break
        if not found: print("Not found {} in dst model".format(src_param[0]))

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

        # list to numpy
        input_img = torch.tensor(input_img, device=device_)
        gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32, device=device_)
        df_boxes = torch.tensor(df_boxes, dtype=torch.float32, device=device_)
        indices = torch.tensor(indices, dtype=torch.int64, device=device_)
        classes = torch.tensor(classes, dtype=torch.int64, device=device_)
        conf_img = torch.tensor(conf_img, device=device_)
        input_seglabel = torch.tensor(input_seglabel, device=device_)

        return input_img, gt_boxes, df_boxes, indices, classes, conf_img, input_seglabel


class MTLoss(nn.Module):
    def __init__(self):
        super(MTLoss, self).__init__()

    def forward(self, Loc, Cls, Seg, gt_box_batch, df_box_batch, idx_batch, cls_batch, bat_s, mining, seg_label):
        if TIMER: start = time.time()
        device_ = "cpu"
        if TIMER: print("Loss_init:", time.time()-start)

        if TIMER: start = time.time()
        if mining:
            # hard negative mining有効時のクラスラベル
            cls_t1 = torch.ones((bat_s, common_params.num_boxes[0], common_params.map_sizes[0], common_params.map_sizes[0]), dtype = torch.int64, device = device_) * -1
            cls_t2 = torch.ones((bat_s, common_params.num_boxes[1], common_params.map_sizes[1], common_params.map_sizes[1]), dtype = torch.int64, device = device_) * -1
            cls_t3 = torch.ones((bat_s, common_params.num_boxes[2], common_params.map_sizes[2], common_params.map_sizes[2]), dtype = torch.int64, device = device_) * -1
            cls_t4 = torch.ones((bat_s, common_params.num_boxes[3], common_params.map_sizes[3], common_params.map_sizes[3]), dtype = torch.int64, device = device_) * -1
            cls_t5 = torch.ones((bat_s, common_params.num_boxes[4], common_params.map_sizes[4], common_params.map_sizes[4]), dtype = torch.int64, device = device_) * -1
            cls_t6 = torch.ones((bat_s, common_params.num_boxes[5], common_params.map_sizes[5], common_params.map_sizes[5]), dtype = torch.int64, device = device_) * -1
        else:
            # hard negative mining無効時のクラスラベル
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

        if TIMER: start = time.time()
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
        if TIMER: print("Loss_forloop:", time.time()-start)


        # 1〜6階層目の教示confidence mapの次元を(バッチ数, DF box数, 高さ, 幅)から(バッチ数, 高さ, 幅, DF box数)に転置
        cls_t1 = cls_t1.permute(0, 2, 3, 1)
        cls_t2 = cls_t2.permute(0, 2, 3, 1)
        cls_t3 = cls_t3.permute(0, 2, 3, 1)
        cls_t4 = cls_t4.permute(0, 2, 3, 1)
        cls_t5 = cls_t5.permute(0, 2, 3, 1)
        cls_t6 = cls_t6.permute(0, 2, 3, 1)

        # 1〜6階層目の教示confidence mapの各次元数を(バッチ数, 高さ, 幅, DF box数)から(バッチ数 * 高さ * 幅 * DF box数)にreshape
        cls_t1 = cls_t1.contiguous()
        cls_t2 = cls_t2.contiguous()
        cls_t3 = cls_t3.contiguous()
        cls_t4 = cls_t4.contiguous()
        cls_t5 = cls_t5.contiguous()
        cls_t6 = cls_t6.contiguous()
        cls_t1 = cls_t1.view(cls_t1.data.shape[0] * cls_t1.data.shape[1] * cls_t1.data.shape[2] * common_params.num_boxes[0])
        cls_t2 = cls_t2.view(cls_t2.data.shape[0] * cls_t2.data.shape[1] * cls_t2.data.shape[2] * common_params.num_boxes[1])
        cls_t3 = cls_t3.view(cls_t3.data.shape[0] * cls_t3.data.shape[1] * cls_t3.data.shape[2] * common_params.num_boxes[2])
        cls_t4 = cls_t4.view(cls_t4.data.shape[0] * cls_t4.data.shape[1] * cls_t4.data.shape[2] * common_params.num_boxes[3])
        cls_t5 = cls_t5.view(cls_t5.data.shape[0] * cls_t5.data.shape[1] * cls_t5.data.shape[2] * common_params.num_boxes[4])
        cls_t6 = cls_t6.view(cls_t6.data.shape[0] * cls_t6.data.shape[1] * cls_t6.data.shape[2] * common_params.num_boxes[5])

        # 1〜6階層目の教示localization mapの次元を(バッチ数, オフセット次元数 * DF box数, 高さ, 幅)から(バッチ数, 高さ, 幅, オフセット次元数 * DF box数)に転置
        loc_t1 = loc_t1.permute(0, 2, 3, 1)
        loc_t2 = loc_t2.permute(0, 2, 3, 1)
        loc_t3 = loc_t3.permute(0, 2, 3, 1)
        loc_t4 = loc_t4.permute(0, 2, 3, 1)
        loc_t5 = loc_t5.permute(0, 2, 3, 1)
        loc_t6 = loc_t6.permute(0, 2, 3, 1)

        # 1〜6階層目の教示localization mapの各次元数を(バッチ数, 高さ, 幅, オフセット次元数 * DF box数)から(バッチ数 * 高さ * 幅 * DF box数, オフセット次元数)にreshape
        loc_t1 = loc_t1.contiguous()
        loc_t2 = loc_t2.contiguous()
        loc_t3 = loc_t3.contiguous()
        loc_t4 = loc_t4.contiguous()
        loc_t5 = loc_t5.contiguous()
        loc_t6 = loc_t6.contiguous()
        loc_t1 = loc_t1.view(loc_t1.data.shape[0] * loc_t1.data.shape[1] * loc_t1.data.shape[2] * common_params.num_boxes[0], int(loc_t1.data.shape[3] / common_params.num_boxes[0]))
        loc_t2 = loc_t2.view(loc_t2.data.shape[0] * loc_t2.data.shape[1] * loc_t2.data.shape[2] * common_params.num_boxes[1], int(loc_t2.data.shape[3] / common_params.num_boxes[1]))
        loc_t3 = loc_t3.view(loc_t3.data.shape[0] * loc_t3.data.shape[1] * loc_t3.data.shape[2] * common_params.num_boxes[2], int(loc_t3.data.shape[3] / common_params.num_boxes[2]))
        loc_t4 = loc_t4.view(loc_t4.data.shape[0] * loc_t4.data.shape[1] * loc_t4.data.shape[2] * common_params.num_boxes[3], int(loc_t4.data.shape[3] / common_params.num_boxes[3]))
        loc_t5 = loc_t5.view(loc_t5.data.shape[0] * loc_t5.data.shape[1] * loc_t5.data.shape[2] * common_params.num_boxes[4], int(loc_t5.data.shape[3] / common_params.num_boxes[4]))
        loc_t6 = loc_t6.view(loc_t6.data.shape[0] * loc_t6.data.shape[1] * loc_t6.data.shape[2] * common_params.num_boxes[5], int(loc_t6.data.shape[3] / common_params.num_boxes[5]))

        # 1〜6階層目の教示confidence mapを結合
        Cls_T = torch.cat([cls_t1, cls_t2, cls_t3, cls_t4, cls_t5, cls_t6], dim = 0)

        # 1〜6階層目の教示localization mapを結合
        Loc_T = torch.cat([loc_t1, loc_t2, loc_t3, loc_t4, loc_t5, loc_t6], dim = 0)

        # loss計算
        x_entropy = nn.CrossEntropyLoss(ignore_index=-1)
        x_entropy_seg = nn.CrossEntropyLoss(ignore_index=-1)
        MSE = nn.MSELoss()
        smooth_L1 = nn.SmoothL1Loss(reduce=False)

        # confidence mapのloss
        loss_cls = x_entropy(Cls.to(device), Cls_T.to(device))
        # localization mapのloss
        #loss_loc = MSE(Loc.to(device), Loc_T.to(device))
        np.set_printoptions(threshold=np.inf)
        loss_loc = smooth_L1(Loc.to(device), Loc_T.to(device))
        print("Loc", Loc.to("cpu").data.shape, Loc.to("cpu").data.numpy())
        print("Loc_T", Loc_T.to("cpu").data.shape, Loc_T.to("cpu").data.numpy())
        print("loss_loc1", loss_loc.to("cpu").data.shape, loss_loc.to("cpu").data.numpy())
        loss_loc = torch.sum(loss_loc, dim=-1)
        print("loss_loc2_sum", loss_loc.to("cpu").data.shape, loss_loc.to("cpu").data.numpy())
        positive_samples = torch.tensor(Cls_T > 0, dtype=torch.float32, device = device)
        print("positive_samples", positive_samples.to("cpu").data.shape, positive_samples.to("cpu").data.numpy())
        loss_loc *= positive_samples
        print("loss_loc3_positives", loss_loc.to("cpu").data.shape, loss_loc.to("cpu").data.numpy())
        n_positives = positive_samples.sum()
        print("n_positives", n_positives)
        loss_loc = torch.sum(loss_loc) / n_positives
        print("loss_loc_finaly", loss_loc)
        exit(1)

        #segmentationのloss
        loss_seg = x_entropy_seg(Seg.to(device), seg_label.to(device))

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

        return loss_cls, loss_loc, loss_seg


class Reporter():
    def __init__(self):
        # Output dir
        today = datetime.datetime.today()
        today_dir = str(today.year) + '-' + str('%02d' % today.month) + '-' + str('%02d' % today.day) + '@' + str('%02d' % today.hour) + '-' + str('%02d' % today.minute) + '-' + str('%02d' % today.second)
        save_dir_suffix = "_PyTorch1"
        self.today_dir_path = path.join(common_params.save_model_dir, today_dir + save_dir_suffix)
        self.loss_dir_path = path.join(self.today_dir_path, "loss")

        if not path.exists(self.today_dir_path):
            os.mkdir(self.today_dir_path)
            os.mkdir(self.loss_dir_path)
        self.save_model_path = path.join(self.today_dir_path, "model")
        self.save_optimizer_path = path.join(self.today_dir_path, "optimizer")
        if not path.exists(self.save_model_path):
            os.mkdir(self.save_model_path)
        if not path.exists(self.save_optimizer_path):
            os.mkdir(self.save_optimizer_path)

        # prepare all log
        file = open(path.join(self.loss_dir_path, "all_log.txt"), 'a')
        out_line = '{} {} {} {} {} {} {} {} {} \n'.format("epoch", "iter", "lr", "loss", "loss_cls", "loss_loc", "loss_seg", "mining", "elapsed_time")
        file.write(out_line)
        file.close()

    def write_single_iter_loss(self, epoch, iter, lr, loss, loss_cls, loss_loc, loss_seg, mining, elapsed_time):
        file = open(path.join(self.loss_dir_path, "loss.txt"), 'a')
        out_line = "{} \n".format(loss.data)
        file.write(out_line)
        file.close()

        file = open(path.join(self.loss_dir_path, "loss_cls.txt"), 'a')
        out_line = "{} \n".format(loss_cls.data)
        file.write(out_line)
        file.close()

        file = open(path.join(self.loss_dir_path, "loss_loc.txt"), 'a')
        out_line = "{} \n".format(loss_loc.data)
        file.write(out_line)
        file.close()

        file = open(path.join(self.loss_dir_path, "loss_seg.txt"), 'a')
        out_line = "{} \n".format(loss_seg.data)
        file.write(out_line)
        file.close()

        file = open(path.join(self.loss_dir_path, "all_log.txt"), 'a')
        out_line = "{} {} {} {} {} {} {} {} {} \n".format(epoch, iter, lr, loss.data, loss_cls.data, loss_loc.data, loss_seg.data, mining, elapsed_time)
        file.write(out_line)
        file.close()

    def write_average_iter_loss(self, epoch, iter, lr, loss, loss_cls, loss_loc, loss_seg, mining, elapsed_time):
        file = open(path.join(self.loss_dir_path, "average_loss.txt"), 'a')
        out_line = "{} {} {} {} {} {} {} {} {} \n".format(epoch, iter, lr, loss.data, loss_cls.data, loss_loc.data, loss_seg.data, mining, elapsed_time)
        file.write(out_line)
        file.close()

    def save_model(self, ssd_model, optimizer, num_epoch, mining):
        suffix = "_with_mining" if mining else "_without_mining"
        torch.save(ssd_model.state_dict(), self.save_model_path + "/SSD_Seg_epoch_" + str(num_epoch + 1) + suffix + ".pth")
        torch.save(optimizer.state_dict(), self.save_optimizer_path + "/SSD_Seg_epoch_" + str(num_epoch + 1) + suffix + "_optim.pth")

def main():
    # SSDNetの読み込み
    ssd_model = SSDNet()

    pretrain_type = args.pretrain_type
    if pretrain_type == 1:
        # 事前学習VGGNetの読み込み
        print("事前学習モデルを読み込んでコピーします")
        vgg_model = VGGNet()
        vgg_b_w = torch.load("./pretrained_model/VGG_ILSVRC_16_layers_fc_reduced.caffemodel.pth")
        del vgg_b_w['fc8.bias'], vgg_b_w['fc8.weight']
        vgg_model.load_state_dict(vgg_b_w)
        copy_model(vgg_model, ssd_model)
        del vgg_model
    elif pretrain_type == 2:
        print("npzファイルから事前学習モデルを読み込んでコピーします")
        copy_model_npz(np.load("../pretrained_model/VGGNet_for_SSD.model"), ssd_model)
    elif pretrain_type == 3:
        print("torchvisionから事前学習モデルを読み込んでコピーします．学習初期は不安定なので，発散する場合は学習率を少し低めにしてください．")
        import torchvision.models as models
        vgg16 = models.vgg16(pretrained=True)
        vgg_table = ["features.0.weight", "features.2.weight", "features.5.weight", "features.7.weight", "features.10.weight", "features.12.weight", "features.14.weight", "features.17.weight", "features.19.weight", "features.21.weight", "features.24.weight", "features.26.weight", "features.28.weight"]
        ssd_table = ["conv1_1.weight", "conv1_2.weight", "conv2_1.weight", "conv2_2.weight", "conv3_1.weight", "conv3_2.weight", "conv3_3.weight", "conv4_1.weight", "conv4_2.weight", "conv4_3.weight", "conv5_1.weight", "conv5_2.weight", "conv5_3.weight"]
        copy_model_torchvision(vgg16, ssd_model, vgg_table, ssd_table)

    if args.gpu >= 0: ssd_model.to(device)

    # 学習再開
    if args.resume:
        ssd_model.load_state_dict(torch.load(args.resume))

    # Dataset import
    train_dataset = MTDataset("./augimg_name_list.txt", "./seglabel_name_list.txt", True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=args.loaderjob)
    # Setup optimizer
    optimizer = optim.SGD(ssd_model.parameters(), lr = common_params.learning_rate, momentum = common_params.momentum, weight_decay = common_params.weight_decay)

    # loss function
    loss_function = MTLoss()

    # loss / model saver
    reporter = Reporter()

    all_iter = 0
    num_report_iter = 50

    begin_at = time.time()
    for num_epoch in tqdm(range(args.epoch)):
        running_loss_cls = 0.0
        running_loss_loc = 0.0
        running_loss_seg = 0.0
        running_loss = 0.0

        # Change learning rate
        for g in optimizer.param_groups:
            if (num_epoch + 1) >= common_params.lr_change_epoch:
                g['lr'] *= common_params.lr_step if g['lr'] >= common_params.lower_lr else common_params.lower_lr
            now_lr = g['lr']

        # epochが4回に1回の割合でhard negative mining有効
        mining = True if (num_epoch + 1) % 4 == 0 else False

        now_datetime = datetime.datetime.today()
        now_datetime = str(now_datetime.year) + '-' + str('%02d' % now_datetime.month) + '-' + str('%02d' % now_datetime.day) + '@' + str('%02d' % now_datetime.hour) + ':' + str('%02d' % now_datetime.minute) + ':' + str('%02d' % now_datetime.second)
        print("\n[Epoch: {} start] Learning rate: {} ({})".format(num_epoch + 1, now_lr, now_datetime))
        if mining: print("[Do HARD NEGATIVE MINING in this epoch!!]")

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
            #ssd_model.zero_grad()
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
            loss_cls, loss_loc, loss_seg = loss_function(Loc, Cls, Seg, gt_box_batch, df_box_batch, idx_batch, cls_batch, args.batchsize, mining, seg_label)
            loss = (loss_cls + loss_loc + loss_seg) / 3.
            if TIMER: print("Loss:", time.time()-start)

            if TIMER: start = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if TIMER: print("Backward:", time.time()-start)

            elapsed_time = datetime.timedelta(seconds = time.time() - begin_at)
            reporter.write_single_iter_loss(num_epoch+1, all_iter, now_lr, loss, loss_cls, loss_loc, loss_seg, mining, elapsed_time)
            sys.stdout.write("\rEpoch: {}, Iter: {}, Loss:{} (cls: {}, loc: {}, seg: {})".format(num_epoch+1, all_iter, loss.data, loss_cls.data, loss_loc.data, loss_seg.data))
            sys.stdout.flush()
            time.sleep(0.01)

            running_loss_cls += loss_cls
            running_loss_loc += loss_loc
            running_loss_seg += loss_seg
            running_loss += loss
            if all_iter % num_report_iter == 0:
                running_loss /= float(num_report_iter)
                running_loss_cls /= float(num_report_iter)
                running_loss_loc /= float(num_report_iter)
                running_loss_seg /= float(num_report_iter)
                reporter.write_average_iter_loss(num_epoch+1, all_iter, now_lr, running_loss, running_loss_cls, running_loss_loc, running_loss_seg, mining, elapsed_time)
                print("\r[E{} I{}] Time: {}, Ave_Loss:{} (c: {}, l: {}, s: {})".format(num_epoch+1, all_iter, elapsed_time, running_loss, running_loss_cls, running_loss_loc, running_loss_seg))
                running_loss_cls = 0.0
                running_loss_loc = 0.0
                running_loss_seg = 0.0
                running_loss = 0.0

        print("\r[Epoch {} done] Save model.".format(num_epoch+1))
        reporter.save_model(ssd_model, optimizer, num_epoch, mining)

    print("\nExit Training\n")

if __name__ == '__main__':
    main()
