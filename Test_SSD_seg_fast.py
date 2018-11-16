#! /usr/bin/env python
# -*- coding: utf-8 -*-
# python Test_SSD_seg_fast.py --indir /Volumes/External/arcdataset/public/ARCdataset_png/test_known/rgb --type '.png'

import numpy as np
import chainer
import chainer.functions as F
from chainer import optimizers
from chainer import cuda
from chainer import Variable
from chainer import serializers
import time
import cPickle
import cv2 as cv
import sys
import math
import argparse

import common_params

from glob import glob
from os import path
import os

# 学習モデルのパス
MODEL_PATH = "./models/DSSD_Seg_epoch_150_without_mining.model"

# WebCamでの検出時に画像を保存する(容量圧迫注意!!) True:する/False:しない
FORCED_SAVE = False

# セグメンテーションのカラー画像生成と画面描画
DISPLAY = True

# FPS表示
FRAMELATE = True

# クラスラベル (クラス名にはスペース(空白)は禁止)
labels = common_params.arc_labels
print(len(labels))

class_color = common_params.arc_class_color[:, ::-1]

parser = argparse.ArgumentParser()
parser.add_argument('--webcam', '-c', type = int, default = -1, help = 'webcam ID / -1 :image file')
parser.add_argument('--indir', '-i', type = str, default = 'none', help = 'input dir')
parser.add_argument('--outdir', '-o', type = str, default = './out/', help = 'output dir of results')
parser.add_argument('--type', '-t', type=str, default='.jpg', help = 'input image type')
parser.add_argument('--gpu', '-g', type = int, default = -1, help = 'GPU ID (negative value indicates CPU)')
args = parser.parse_args()

IN_DIR = args.indir
OUT_DIR = args.outdir
IN_TYPE = args.type
OUT_TYPE = '.png'

if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np


def init():
    # エラーチェック
    save_flag = (args.webcam < 0) or FORCED_SAVE # WebCamモードオフまたは強制保存オンで保存フラグ
    error_flag = False
    if not(os.path.exists(MODEL_PATH)):
        print('[Error] 学習モデルが見つかりません: ' + MODEL_PATH)
        error_flag = True
    if (len(glob(OUT_DIR + '/detection/*')) != 0) and save_flag:
        print('[Error] 前回の結果が' + OUT_DIR + 'に残っています．上書き防止のため停止します．')
        error_flag = True
    if (args.webcam >= 0) and (IN_DIR != 'none'):
        print('[Error] WebCamを使用しながらファイル入力はできません．')
        error_flag = True
    if (args.webcam < 0 ) and (IN_DIR == 'none'):
        print('[Error] 入力ファイルを指定するか，--webcam <ID> でWebCamを使って下さい．')
        error_flag = True

    # ディレクトリの作成
    if not(error_flag) and save_flag:
        make_dir_list = [
            OUT_DIR,
            OUT_DIR + '/detection',                     # 検出結果のb-box画像
            OUT_DIR + '/segmentation_results',          # セグメンテーション結果
            OUT_DIR + '/segmentation_results_large',    # セグメンテーション結果(入力サイズに拡大)
            OUT_DIR + '/segmentation_results_gray',     # セグメンテーション結果(グレースケール)
            OUT_DIR + '/detection_txt'                  # 検出結果のb-boxテキスト
        ]
        for dirname in make_dir_list:
            if not(os.path.exists(dirname)): os.mkdir(dirname)

    return error_flag


# confidence mapのsoftmaxを計算
def mboxSoftmax(confidence_maps, num_classes, num_boxes):

    s = np.zeros((confidence_maps.shape[0], confidence_maps.shape[1], confidence_maps.shape[2]), np.float32)

    bs = 0
    be = num_classes
    for b in xrange(0, num_boxes):

        t = confidence_maps[bs : be, :, :]

        total = 0
        for i in xrange(0, t.shape[0]):
            total += np.exp(t[i, :, :])

        for i in xrange(0, t.shape[0]):
            s[bs + i, :, :] = np.exp(t[i, :, :]) / total

        bs = be
        be += num_classes

    return s


# LocとClsをCPUで扱える形式に変換
def to_CPU(Loc, Cls):
    Loc = cuda.to_cpu(Loc.data)
    Cls = cuda.to_cpu(Cls.data)

    return Loc, Cls


# クラス確率の高いdefault boxを検出
def multiBoxDetection(cls_score_maps, localization_maps, num_dbox, num_class, offset_dim, min_size, max_size, step, aspect_ratio):

    box_offsets = []
    default_boxes = []
    class_labels = []
    class_scores = []

    img_width = common_params.insize
    img_height = common_params.insize

    map_size = cls_score_maps.shape[1] * cls_score_maps.shape[2]
    for i in xrange(0, map_size):

        c = int(i % cls_score_maps.shape[1])
        r = int(i / cls_score_maps.shape[1])

        mbox_max_val = 0
        mbox_max_idx = 0
        mbox_num = 0

        bs = 0
        be = num_class
        for b in xrange(0, num_dbox):

            max_val = np.max(cls_score_maps[bs : be, r, c])
            max_idx = int(np.argmax(cls_score_maps[bs : be, r, c]))

            if max_val > mbox_max_val and max_idx != 0:
                mbox_max_val = max_val
                mbox_max_idx = max_idx
                mbox_num = b

            bs = be
            be += num_class

        bs = mbox_num * offset_dim
        be = bs + offset_dim
        b_offset = localization_maps[bs : be, r, c]

        offset_ = 0.5

        if mbox_max_val >= 0.7:
            center_x = float((c + offset_) * step)
            center_y = float((r + offset_) * step)

            if mbox_num == 0:
                box_width = box_height = min_size
                xmin = (center_x - box_width / 2.) / img_width
                ymin = (center_y - box_height / 2.) / img_height
                xmax = (center_x + box_width / 2.) / img_width
                ymax = (center_y + box_height / 2.) / img_height
            elif mbox_num == 1:
                box_width = box_height = np.sqrt(min_size * max_size)
                xmin = (center_x - box_width / 2.) / img_width
                ymin = (center_y - box_height / 2.) / img_height
                xmax = (center_x + box_width / 2.) / img_width
                ymax = (center_y + box_height / 2.) / img_height
            elif mbox_num == 2:
                box_width = min_size * np.sqrt(float(aspect_ratio[0]))
                box_height = min_size / np.sqrt(float(aspect_ratio[0]))
                xmin = (center_x - box_width / 2.) / img_width
                ymin = (center_y - box_height / 2.) / img_height
                xmax = (center_x + box_width / 2.) / img_width
                ymax = (center_y + box_height / 2.) / img_height
            elif mbox_num == 3:
                box_width = min_size * np.sqrt(1. / float(aspect_ratio[0]))
                box_height = min_size / np.sqrt(1. / float(aspect_ratio[0]))
                xmin = (center_x - box_width / 2.) / img_width
                ymin = (center_y - box_height / 2.) / img_height
                xmax = (center_x + box_width / 2.) / img_width
                ymax = (center_y + box_height / 2.) / img_height
            elif mbox_num == 4:
                box_width = min_size * np.sqrt(float(aspect_ratio[1]))
                box_height = min_size / np.sqrt(float(aspect_ratio[1]))
                xmin = (center_x - box_width / 2.) / img_width
                ymin = (center_y - box_height / 2.) / img_height
                xmax = (center_x + box_width / 2.) / img_width
                ymax = (center_y + box_height / 2.) / img_height
            elif mbox_num == 5:
                box_width = min_size * np.sqrt(1. / float(aspect_ratio[1]))
                box_height = min_size / np.sqrt(1. / float(aspect_ratio[1]))
                xmin = (center_x - box_width / 2.) / img_width
                ymin = (center_y - box_height / 2.) / img_height
                xmax = (center_x + box_width / 2.) / img_width
                ymax = (center_y + box_height / 2.) / img_height

            box_offsets.append(b_offset)
            default_boxes.append([min(max(xmin, 0.), 1.), min(max(ymin, 0.), 1.), min(max(xmax, 0.), 1.), min(max(ymax, 0.), 1.), mbox_num])
            class_labels.append(mbox_max_idx)
            class_scores.append(mbox_max_val)

    return (box_offsets, default_boxes, class_labels, class_scores)

# bounding box候補を検出 (default boxの補正)
def candidatesDetection(offsets, default_boxes, class_labels, class_scores, num_classes, color_img, variance):

    img_width = color_img.shape[1]
    img_height = color_img.shape[0]

    candidates = []
    for i in xrange(0, num_classes):
        candidates.append([])

    for det in xrange(0, len(class_labels)):

        pred_xmin = (default_boxes[det][0] + offsets[det][0] * variance)
        pred_ymin = (default_boxes[det][1] + offsets[det][1] * variance)
        pred_xmax = (default_boxes[det][2] + offsets[det][2] * variance)
        pred_ymax = (default_boxes[det][3] + offsets[det][3] * variance)

        # pred_xmin = default_boxes[det][0]
        # pred_ymin = default_boxes[det][1]
        # pred_xmax = default_boxes[det][2]
        # pred_ymax = default_boxes[det][3]

        pred_xmin = min(max(pred_xmin, 0.), 1.) * img_width
        pred_ymin = min(max(pred_ymin, 0.), 1.) * img_height
        pred_xmax = min(max(pred_xmax, 0.), 1.) * img_width
        pred_ymax = min(max(pred_ymax, 0.), 1.) * img_height
        candidates[class_labels[det]].append([pred_xmin, pred_ymin, pred_xmax, pred_ymax, class_scores[det]])
        #print(class_labels)
    return candidates

# 同じクラスじ属するbounding boxの重なり率を計算
def jaccardOverlap(bbox1, bbox2):

    if (bbox2[0] > bbox1[2]) or (bbox2[2] < bbox1[0]) or (bbox2[1] > bbox1[3]) or (bbox2[3] < bbox1[1]):
        overlap = 0.
    else:
        inter_xmin = max(bbox1[0], bbox2[0])
        inter_ymin = max(bbox1[1], bbox2[1])
        inter_xmax = min(bbox1[2], bbox2[2])
        inter_ymax = min(bbox1[3], bbox2[3])

        inter_width = inter_xmax - inter_xmin
        inter_height = inter_ymax - inter_ymin
        inter_size = inter_width * inter_height

        bbox1_size = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_size = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        overlap = inter_size / (bbox1_size + bbox2_size - inter_size)

    return overlap

def seg_jaccardOverlap(bbox1, bbox2):

    x_max = bbox2[0] + bbox2[2]
    y_max = bbox2[1] + bbox2[3]
    if (bbox2[0] > bbox1[2]) or (x_max < bbox1[0]) or (bbox2[1] > bbox1[3]) or (y_max < bbox1[1]):
        overlap = 0.
    else:
        x_max = bbox2[0] + bbox2[2]
        y_max = bbox2[1] + bbox2[3]
        inter_xmin = max(bbox1[0], bbox2[0])
        inter_ymin = max(bbox1[1], bbox2[1])
        inter_xmax = min(bbox1[2], x_max)
        inter_ymax = min(bbox1[3], y_max)

        inter_width = inter_xmax - inter_xmin
        inter_height = inter_ymax - inter_ymin
        inter_size = inter_width * inter_height

        bbox1_size = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_size = (x_max - bbox2[0]) * (y_max - bbox2[1])

        overlap = inter_size / (bbox1_size + bbox2_size - inter_size)

    return overlap


# bounding box候補のnon-maximum suppresion
def nonMaximumSuppresion(candidates, sbox):

    overlap_th = 0.2

    for i in xrange(0, len(candidates)):
        box_num = len(candidates[i])

        for j in xrange(0, box_num):
            for s in xrange(0, len(sbox)):
                if i == sbox[s][4]:
                    #print(sbox[s][4])
                    seg_overlap = seg_jaccardOverlap(candidates[i][j], sbox[s])
                    #print(candidates[i][js][4])
                    seg_overlap = (seg_overlap + candidates[i][j][4]) / 2
                    candidates[i][j][4] = seg_overlap
                    #print(seg_overlap)


    for i in xrange(0, len(candidates)):

        box_num = len(candidates[i])

        js = 0
        for j in xrange(0, box_num):
            ks = js + 1
            for k in xrange(j + 1, box_num):

                if ks >= len(candidates[i]) or js >= len(candidates[i]):
                    continue

                overlap = jaccardOverlap(candidates[i][js], candidates[i][ks])

                #for s in xrange(len(sbox)):
                #    if i == sbox[s][4]:
                    #    print(sbox[s][4])
                    #    seg_overlap = seg_jaccardOverlap(candidates[i][js], sbox[s])
                    #    #print(candidates[i][js][4])
                    #    seg_overlap = (seg_overlap + candidates[i][js][4]) / 2
                    #    candidates[i][js][4] = seg_overlap
                        #print(seg_overlap)
                #print(overlap)

                if overlap >= overlap_th and candidates[i][js][4] >= candidates[i][ks][4]:
                    candidates[i].pop(ks)
                    ks -= 1
                elif overlap >= overlap_th and candidates[i][js][4] < candidates[i][ks][4]:
                    candidates[i][js], candidates[i][ks] = candidates[i][ks], candidates[i][js]
                    candidates[i].pop(ks)
                    ks -= 1

                ks += 1
            js += 1

    return candidates

# box検出結果を画像に描画してテキストとともに保存
def saveDetection(final_detections, out_img, filename):
    save_flag = (args.webcam < 0) or FORCED_SAVE # WebCamモードオフまたは強制保存オンで保存フラグ

    # 検出したbounding boxを画像に描画
    offset_ = 0.5
    font = cv.FONT_HERSHEY_SIMPLEX
    for i in xrange(0, len(final_detections)):
            class_name = labels[i]
            color = class_color[i]
            for j in xrange(0, len(final_detections[i])):
                p1 = int(final_detections[i][j][0] + offset_)
                p2 = int(final_detections[i][j][1] + offset_)
                p3 = int(final_detections[i][j][2] + offset_)
                p4 = int(final_detections[i][j][3] + offset_)
                cv.rectangle(out_img, (p1, p2), (p3, p4), (color[0], color[1], color[2]), 5)
                q1 = p1
                q2 = p4
                if False: # ID表示
                    cv.rectangle(out_img, (q1, q2 - 40), (q1 + 150, q2), (color[0], color[1], color[2]), -1)
                    cv.putText(out_img, str(i) + ": " + str(('%.2f' % final_detections[i][j][4])), (q1, q2 - 8), font, 1, (0, 0, 0), 2, cv.CV_AA)
                    cv.putText(out_img, str(i) + ": " + str(('%.2f' % final_detections[i][j][4])), (q1, q2 - 8), font, 1, (255, 255, 255), 1, cv.CV_AA)
                if True: # クラス名表示
                    cv.rectangle(out_img, (q1, q2 - 30), (q1 + len(class_name)*25, q2), (color[0], color[1], color[2]), -1)
                    cv.putText(out_img, str(i) + " " + class_name + ": " + str(('%.2f' % final_detections[i][j][4])), (q1, q2 - 8), font, 0.7, (0, 0, 0), 2, cv.CV_AA)
                    cv.putText(out_img, str(i) + " " + class_name + ": " + str(('%.2f' % final_detections[i][j][4])), (q1, q2 - 8), font, 0.7, (255, 255, 255), 1, cv.CV_AA)

                #print("p: {} {}".format(str(i), class_name))

                #検出したクラスのラベル、スコア、座標を出力
                if save_flag:
                    f = open(OUT_DIR + '/detection_txt/' + filename + 'res.txt', 'a')
                    f.write("{} {} {} {} {} {}\n".format(str(i), final_detections[i][j][4], p1, p2, p3, p4))
                    f.close()
    # 画像保存
    if save_flag:
        cv.imwrite(OUT_DIR + '/detection/' + filename + OUT_TYPE, out_img)

    #cv.namedWindow("Final Detections", cv.WINDOW_NORMAL)
    #out_img_small = cv.resize(out_img, (int(out_img.shape[1] * 0.8), int(out_img.shape[0] * 0.8)))

    return out_img

def detection(img, ssd_model, filename, min_sizes, max_sizes):
    # タイマーリセット
    total_time = 0.0
    processing_time = 0.0
    drawing_time = 0.0
    fps_start_time = time.time()

    save_flag = (args.webcam < 0) or FORCED_SAVE # WebCamモードオフまたは強制保存オンで保存フラグ
    classes = []
    fps_start_time = time.time()

    # 入力画像をSSDの入力サイズにリサイズ
    start = time.time()
    input_img = cv.resize(img, (common_params.insize, common_params.insize), interpolation = cv.INTER_CUBIC)
    out_img = img.copy()

    input_img = input_img.astype(np.float32)
    input_img -= np.array([103.939, 116.779, 123.68])
    input_img = input_img.transpose(2, 0, 1)
    # mean_val = [104., 117, 123]
    # input_img[0, :, :] -= mean_val[0]
    # input_img[1, :, :] -= mean_val[1]
    # input_img[2, :, :] -= mean_val[2]
    input_data = []
    input_data.append(input_img)

    x_data = Variable(xp.asarray(input_data, np.float32))
    ssd_model.train = False
    elapsed_time = time.time() - start
    print ('Resize : ', elapsed_time)
    processing_time += elapsed_time
    total_time += elapsed_time

    # SSDのforward
    start = time.time()
    Loc1, Cls1, Loc2, Cls2, Loc3, Cls3, Loc4, Cls4, Loc5, Cls5, Loc6, Cls6, Seg = ssd_model(x_data)
    elapsed_time = time.time() - start
    print ('SSD_forward : ', elapsed_time)
    processing_time += elapsed_time
    total_time += elapsed_time

    # CPUで処理
    start = time.time()
    Loc1, Cls1 = to_CPU(Loc1, Cls1)
    Loc2, Cls2 = to_CPU(Loc2, Cls2)
    Loc3, Cls3 = to_CPU(Loc3, Cls3)
    Loc4, Cls4 = to_CPU(Loc4, Cls4)
    Loc5, Cls5 = to_CPU(Loc5, Cls5)
    Loc6, Cls6 = to_CPU(Loc6, Cls6)
    elapsed_time = time.time() - start
    print ('to_CPU : ', elapsed_time)
    processing_time += elapsed_time
    total_time += elapsed_time

    # セグメンテーション推論
    start = time.time()
    if False:
        # 推論時にSoftmaxを用いる（低速）
        pred = F.softmax(Seg)
        pred.to_cpu()
        seg_result = np.squeeze(pred.data[0,:,:,:])
        seg_result_max = np.argmax(seg_result, axis=0)
    else:
        # Softmaxなし（高速）
        Seg.to_cpu()
        pred = np.argmax(Seg.data, axis=1)
        seg_result_max = np.squeeze(pred[0,:,:])

    # セグメンテーション結果をグレー画像でもっておく(保存用/外接矩形抽出用)
    gray = seg_result_max.astype(np.uint8)
    gray_large = cv.resize(gray, (1280, 960), interpolation = cv.INTER_NEAREST)
    elapsed_time = time.time() - start
    print ('seg_estimate : ', elapsed_time)
    processing_time += elapsed_time
    total_time += elapsed_time

    # セグメンテーション保存
    start = time.time()
    if DISPLAY:
        # セグメンテーション結果をカラーに戻す
        rgb_r = seg_result_max.copy()
        rgb_g = seg_result_max.copy()
        rgb_b = seg_result_max.copy()

        for k in range(0, class_color.shape[0]):
            rgb_r[seg_result_max==k] = class_color[k,0]
            rgb_g[seg_result_max==k] = class_color[k,1]
            rgb_b[seg_result_max==k] = class_color[k,2]

        rgb = np.zeros((seg_result_max.shape[0], seg_result_max.shape[1], 3))
        rgb[:,:,0] = rgb_r
        rgb[:,:,1] = rgb_g
        rgb[:,:,2] = rgb_b

        rgb = rgb.astype(np.uint8)
        if save_flag: cv.imwrite(OUT_DIR + '/segmentation_results/' + filename + OUT_TYPE, rgb)
        rgb = cv.resize(rgb, (1280, 960), interpolation = cv.INTER_NEAREST)
        if save_flag: cv.imwrite(OUT_DIR + '/segmentation_results_large/' + filename + OUT_TYPE, rgb)
        if save_flag: cv.imwrite(OUT_DIR + '/segmentation_results_gray/' + filename + OUT_TYPE, gray)

        rgb_s = rgb.copy()
        elapsed_time = time.time() - start
        print ('seg_save : ', elapsed_time)
        drawing_time += elapsed_time
        total_time += elapsed_time

    sbox = []

    # セグメンテーションの外接矩形を取得
    start = time.time()
    #for g in xrange(0, len(classes)): # OLD
    for g in xrange(1, len(labels)+1):
        gray_extract = gray_large.copy()
        #gray_extract[gray_extract != classes[g]] = 0 # OLD
        gray_extract[gray_extract != g] = 0
        #ret, thresh = cv.threshold(gray_extract, classes[g]-1, 255, 0) # OLD
        ret, thresh = cv.threshold(gray_extract, g-1, 255, 0)
        contours, hierarchy = cv.findContours(thresh, 1, 2)

        max_contours = 0
        max_i = 0
        for i in xrange(len(contours)):
            if max_contours <= len(contours[i]):
                max_contours = len(contours[i])
                max_i = i

        if len(contours) != 0:
            x,y,w,h = cv.boundingRect(contours[max_i])
            if DISPLAY: cv.rectangle(rgb_s,(x,y),(x+w,y+h),(0,255,0),2)
            #sbox.append([x, y, w, h, classes[g]]) # OLD
            sbox.append([x, y, w, h, g])

            #cv.imshow('image', rgb_s)
            #cv.waitKey()
            #cv.destroyAllWindows()
    elapsed_time = time.time() - start
    print ('b-box extract : ', elapsed_time)
    processing_time += elapsed_time
    total_time += elapsed_time

    # 各階層のconfidence mapのsoftmaxを計算
    start = time.time()
    cls_score1 = mboxSoftmax(Cls1[0], common_params.num_of_classes, common_params.num_boxes[0])
    cls_score2 = mboxSoftmax(Cls2[0], common_params.num_of_classes, common_params.num_boxes[1])
    cls_score3 = mboxSoftmax(Cls3[0], common_params.num_of_classes, common_params.num_boxes[2])
    cls_score4 = mboxSoftmax(Cls4[0], common_params.num_of_classes, common_params.num_boxes[3])
    cls_score5 = mboxSoftmax(Cls5[0], common_params.num_of_classes, common_params.num_boxes[4])
    cls_score6 = mboxSoftmax(Cls6[0], common_params.num_of_classes, common_params.num_boxes[5])
    elapsed_time = time.time() - start
    print ('softmax : ', elapsed_time)
    processing_time += elapsed_time
    total_time += elapsed_time

    # クラス確率の高いdefault boxの検出
    start = time.time()
    offsets1, default_boxes1, class_labels1, class_scores1 = multiBoxDetection(cls_score1, Loc1[0], common_params.num_boxes[0], common_params.num_of_classes, common_params.num_of_offset_dims, min_sizes[0], max_sizes[0], common_params.steps[0], common_params.aspect_ratios[0])
    offsets2, default_boxes2, class_labels2, class_scores2 = multiBoxDetection(cls_score2, Loc2[0], common_params.num_boxes[1], common_params.num_of_classes, common_params.num_of_offset_dims, min_sizes[1], max_sizes[1], common_params.steps[1], common_params.aspect_ratios[1])
    offsets3, default_boxes3, class_labels3, class_scores3 = multiBoxDetection(cls_score3, Loc3[0], common_params.num_boxes[2], common_params.num_of_classes, common_params.num_of_offset_dims, min_sizes[2], max_sizes[2], common_params.steps[2], common_params.aspect_ratios[2])
    offsets4, default_boxes4, class_labels4, class_scores4 = multiBoxDetection(cls_score4, Loc4[0], common_params.num_boxes[3], common_params.num_of_classes, common_params.num_of_offset_dims, min_sizes[3], max_sizes[3], common_params.steps[3], common_params.aspect_ratios[3])
    offsets5, default_boxes5, class_labels5, class_scores5 = multiBoxDetection(cls_score5, Loc5[0], common_params.num_boxes[4], common_params.num_of_classes, common_params.num_of_offset_dims, min_sizes[4], max_sizes[4], common_params.steps[4], common_params.aspect_ratios[4])
    offsets6, default_boxes6, class_labels6, class_scores6 = multiBoxDetection(cls_score6, Loc6[0], common_params.num_boxes[5], common_params.num_of_classes, common_params.num_of_offset_dims, min_sizes[5], max_sizes[5], common_params.steps[5], common_params.aspect_ratios[5])
    elapsed_time = time.time() - start
    print ('multibox_detection : ', elapsed_time)
    processing_time += elapsed_time
    total_time += elapsed_time

    # オフセットベクトルによりdefault boxを補正
    start = time.time()
    candidates1 = candidatesDetection(offsets1, default_boxes1, class_labels1, class_scores1, common_params.num_of_classes, img, common_params.loc_var)
    candidates2 = candidatesDetection(offsets2, default_boxes2, class_labels2, class_scores2, common_params.num_of_classes, img, common_params.loc_var)
    candidates3 = candidatesDetection(offsets3, default_boxes3, class_labels3, class_scores3, common_params.num_of_classes, img, common_params.loc_var)
    candidates4 = candidatesDetection(offsets4, default_boxes4, class_labels4, class_scores4, common_params.num_of_classes, img, common_params.loc_var)
    candidates5 = candidatesDetection(offsets5, default_boxes5, class_labels5, class_scores5, common_params.num_of_classes, img, common_params.loc_var)
    candidates6 = candidatesDetection(offsets6, default_boxes6, class_labels6, class_scores6, common_params.num_of_classes, img, common_params.loc_var)
    elapsed_time = time.time() - start
    print ('localization : ', elapsed_time)
    processing_time += elapsed_time
    total_time += elapsed_time

    # 各階層のbounding box候補を統合
    start = time.time()
    all_candidate = []
    for i in xrange(0, common_params.num_of_classes):
        all_candidate.append([])
        all_candidate[i].extend(candidates1[i])
        all_candidate[i].extend(candidates2[i])
        all_candidate[i].extend(candidates3[i])
        all_candidate[i].extend(candidates4[i])
        all_candidate[i].extend(candidates5[i])
        all_candidate[i].extend(candidates6[i])
    #print(all_candidate)
    elapsed_time = time.time() - start
    print ('candidate_extend : ', elapsed_time)
    processing_time += elapsed_time
    total_time += elapsed_time

    # non-maximum suppresionによりbounding boxの最終結果を出力
    start = time.time()
    final_detections = nonMaximumSuppresion(all_candidate, sbox)
    #final_detections = all_candidate
    elapsed_time = time.time() - start
    print ('non-max_supp : ', elapsed_time)
    processing_time += elapsed_time
    total_time += elapsed_time

    # 画像保存
    start = time.time()
    out_img = saveDetection(final_detections, out_img, filename)
    elapsed_time = time.time() - start
    print ('save detection : ', elapsed_time)
    drawing_time += elapsed_time
    total_time += elapsed_time

    fps_end_time = time.time()
    fps = 1 / total_time
    print ('Total : ', total_time)
    print ('Processing Time : ', processing_time)
    print ('Drawing Time : ', drawing_time)
    print('FPS: ', fps)
    print('sec2: ', fps_end_time - fps_start_time)

    # 結果を画面に出力
    if DISPLAY:
        out_img_half = cv.resize(out_img, (int(out_img.shape[1] * 0.9), int(out_img.shape[0] * 0.9)))
        out_seg_half = cv.resize(rgb_s, (int(out_img.shape[1] * 0.9), int(out_img.shape[0] * 0.9)))
        out_final= cv.hconcat([out_img_half, out_seg_half])
        #rgb_half = cv.resize(rgb, (int(rgb.shape[1] * 0.8), int(rgb.shape[0] * 0.8)))
        # FPS表示
        if FRAMELATE:
            cv.rectangle(out_final, (0, 0), (180, 35), (0, 0, 0), -1)
            cv.putText(out_final, "FPS:" + "{0:.2f}".format(fps), (0, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.CV_AA)
        cv.imshow("Final Detections", out_final)
        cv.waitKey(1)

    print('----- Detection Done -----')

def main():
    if FORCED_SAVE: print('[Warn] 強制保存がオンになっているためWebCam動作中でも検出結果を保存します．ストレージの容量に注意！')

    # エラーチェック
    if init():
        sys.exit(1)

    # 学習モデル読み込み
    print('[Info] SSD Netの読み込み中...')
    from SSD_seg_Net import SSDNet
    ssd_model = SSDNet()
    serializers.load_npz(MODEL_PATH, ssd_model)
    print('-> 読み込み完了')

    # CUDA INIT
    if args.gpu >= 0:
        print('[Info] CUDAを使います．GPU ID: ' + str(args.gpu))
        cuda.get_device(args.gpu).use()
        ssd_model.to_gpu()

    # default boxのサイズリスト計算
    step = int(math.floor((common_params.max_ratio - common_params.min_ratio) / (len(common_params.mbox_source_layers) - 2)))
    min_sizes = []
    max_sizes = []
    for ratio in xrange(common_params.min_ratio, common_params.max_ratio + 1, step):
            min_sizes.append(common_params.insize * ratio / 100.)
            max_sizes.append(common_params.insize * (ratio + step) / 100.)
    min_sizes = [common_params.insize * 10 / 100.] + min_sizes
    max_sizes = [common_params.insize * 20 / 100.] + max_sizes

    # 入力画像の読み込み
    if args.webcam >= 1:
        # Webcamから画像入力
        cap = cv.VideoCapture(args.webcam)
        while True:
            if FRAMELATE: start_time = time.time()
            for i in xrange(0,5):
                ret, img = cap.read()
            if img is None:
                print('[Error] WebCamから画像が取得できません')
                sys.exit(1)
            detection(img, ssd_model, 'webcam', min_sizes, max_sizes)

    else:
        # Webcamじゃない場合は画像リスト読み込み
        color_img = glob(path.join(IN_DIR, '*' + IN_TYPE))
        color_img.sort()

        # 読み込んだリストを順次検出
        for lf in xrange(len(color_img)):
            img = cv.imread(color_img[lf])
            filename, ext = os.path.splitext(os.path.basename(color_img[lf]))

            if img is None:
                print('[Error] 画像が読み込めません: ' + str(color_img[lf]))
                sys.exit(1)
            detection(img, ssd_model, filename, min_sizes, max_sizes)


if __name__ == '__main__':
    main()
