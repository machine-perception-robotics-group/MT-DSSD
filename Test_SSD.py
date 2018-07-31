#coding: utf-8
import numpy as np
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
from glob import glob
from os import path
import os

import common_params

# クラスラベル (クラス名にはスペース(空白)は禁止)
labels = [
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
print(len(labels))

class_color = np.array([
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

class_color = class_color[:, ::-1]

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='./')
parser.add_argument('--out', type=str, default='./')
parser.add_argument('--type', type=str, default='.jpg')
parser.add_argument('--gpu', '-g', default = -1, type = int, help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np


# confidence mapのsoftmaxを計算
def mboxSoftmax(confidence_maps, num_classes, num_boxes):

    s = xp.zeros((confidence_maps.shape[0], confidence_maps.shape[1], confidence_maps.shape[2]), np.float32)

    bs = 0
    be = num_classes
    for b in xrange(0, num_boxes):

        t = confidence_maps[bs : be, :, :]

        total = 0
        for i in xrange(0, t.shape[0]):
            total += xp.exp(t[i, :, :])

        for i in xrange(0, t.shape[0]):
            s[bs + i, :, :] = xp.exp(t[i, :, :]) / total

        bs = be
        be += num_classes

    return s

# クラス確率の高いdefault boxを検出
def multiBoxDetection(cls_score_maps, localization_maps, num_dbox, min_size, max_size, step, aspect_ratio):

    box_offsets = []
    default_boxes = []
    class_labels = []
    class_scores = []


    num_class = common_params.num_of_classes
    offset_dim = common_params.num_of_offset_dims

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
        #cls_max_val = np.max(cls_score_maps[bs : be, r, c])
        #cls_max_idx = int(np.argmax(cls_score_maps[bs : be, r, c]))

        offset_ = 0.5

        if mbox_max_val >= 0.5:
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
            #class_labels.append(cls_max_idx)
            #class_scores.append(cls_max_val)
            class_labels.append(mbox_max_idx)
            class_scores.append(mbox_max_val)


    return (box_offsets, default_boxes, class_labels, class_scores)

# bounding box候補を検出 (default boxの補正)
def candidatesDetection(offsets, default_boxes, class_labels, class_scores, color_img):

    img_width = color_img.shape[1]
    img_height = color_img.shape[0]

    candidates = []

    for det in xrange(0, len(class_labels)):

        pred_xmin = (default_boxes[det][0] + offsets[det][0] * common_params.loc_var)
        pred_ymin = (default_boxes[det][1] + offsets[det][1] * common_params.loc_var)
        pred_xmax = (default_boxes[det][2] + offsets[det][2] * common_params.loc_var)
        pred_ymax = (default_boxes[det][3] + offsets[det][3] * common_params.loc_var)

        # pred_xmin = default_boxes[det][0]
        # pred_ymin = default_boxes[det][1]
        # pred_xmax = default_boxes[det][2]
        # pred_ymax = default_boxes[det][3]

        pred_xmin = min(max(pred_xmin, 0.), 1.) * img_width
        pred_ymin = min(max(pred_ymin, 0.), 1.) * img_height
        pred_xmax = min(max(pred_xmax, 0.), 1.) * img_width
        pred_ymax = min(max(pred_ymax, 0.), 1.) * img_height
        candidates.append([pred_xmin, pred_ymin, pred_xmax, pred_ymax, class_labels[det], class_scores[det]])

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

# bounding box候補のnon-maximum suppresion

def nonMaximumSuppresion(candidates):

    overlap_th = 0.3

    box_num = len(candidates)

    js = 0
    for j in xrange(0, box_num):
        ks = js + 1
        for k in xrange(j + 1, box_num):

            if ks >= len(candidates) or js >= len(candidates):
                continue

            #if candidates[js][4] != candidates[ks][4]:
                #continue

            overlap = jaccardOverlap(candidates[js], candidates[ks])

            if overlap >= overlap_th and candidates[js][5] >= candidates[ks][5]:
                candidates.pop(ks)
                ks -= 1
            elif overlap >= overlap_th and candidates[js][5] < candidates[ks][5]:
                candidates[js], candidates[ks] = candidates[ks], candidates[js]
                candidates.pop(ks)
                ks -= 1

            ks += 1
        js += 1

    return candidates




input_dir = args.dir
input_type = args.type

output_dir = args.out
output_type = input_type

if not path.exists(output_dir):
    os.mkdir(output_dir)



print('SSD Netの読み込み中...')
from SSD_Net import SSDNet
ssd_model = SSDNet()
serializers.load_npz('./models/SSD_epoch_41_without_mining.model', ssd_model)
print('-> 読み込み完了')


if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    ssd_model.to_gpu()

step = int(math.floor((common_params.max_ratio - common_params.min_ratio) / (len(common_params.mbox_source_layers) - 2)))

min_sizes = []
max_sizes = []

for ratio in xrange(common_params.min_ratio, common_params.max_ratio + 1, step):
    min_sizes.append(common_params.insize * ratio / 100.)
    max_sizes.append(common_params.insize * (ratio + step) / 100.)

min_sizes = [common_params.insize * 10 / 100.] + min_sizes
max_sizes = [common_params.insize * 20 / 100.] + max_sizes

cv.namedWindow("Final Detections", cv.WINDOW_AUTOSIZE)

img_list = glob(path.join(input_dir, '*'))

for ls in img_list:

    print ls

    items = ls.split('/')
    file_name, ext = os.path.splitext(items[-1])


    color_img = cv.imread(input_dir + '/' + file_name + input_type, cv.IMREAD_COLOR)

    """ 画像が読み込めなかった場合 """
    if color_img is None:
        print('画像が読み込めません')
        sys.exit(1)

    # 入力画像をSSDの入力サイズにリサイズ
    input_img = cv.resize(color_img, (common_params.insize, common_params.insize), interpolation = cv.INTER_CUBIC)

    out_cls = color_img.copy()

    input_img = input_img.astype(np.float32)
    input_img -= np.array([103.939, 116.779, 123.68])
    input_img = input_img.transpose(2, 0, 1)


    input_data = []
    input_data.append(input_img)

    x_data = Variable(xp.asarray(input_data, np.float32))

    ssd_model.train = False

    total_time = 0.0

    # SSDのforward
    Loc1, Cls1, Loc2, Cls2, Loc3, Cls3, Loc4, Cls4, Loc5, Cls5, Loc6, Cls6 = ssd_model(x_data)


    # 各階層のconfidence mapのsoftmaxを計算
    cls_score1 = mboxSoftmax(Cls1.data[0], common_params.num_of_classes, common_params.num_boxes[0])
    cls_score2 = mboxSoftmax(Cls2.data[0], common_params.num_of_classes, common_params.num_boxes[1])
    cls_score3 = mboxSoftmax(Cls3.data[0], common_params.num_of_classes, common_params.num_boxes[2])
    cls_score4 = mboxSoftmax(Cls4.data[0], common_params.num_of_classes, common_params.num_boxes[3])
    cls_score5 = mboxSoftmax(Cls5.data[0], common_params.num_of_classes, common_params.num_boxes[4])
    cls_score6 = mboxSoftmax(Cls6.data[0], common_params.num_of_classes, common_params.num_boxes[5])

    # Classスコアの高いdefault boxの検出
    offsets1, default_boxes1, class_labels1, class_scores1 = multiBoxDetection(cls_score1, Loc1.data[0], common_params.num_boxes[0], min_sizes[0], max_sizes[0], common_params.steps[0], common_params.aspect_ratios[0])
    offsets2, default_boxes2, class_labels2, class_scores2 = multiBoxDetection(cls_score2, Loc2.data[0], common_params.num_boxes[1], min_sizes[1], max_sizes[1], common_params.steps[1], common_params.aspect_ratios[1])
    offsets3, default_boxes3, class_labels3, class_scores3 = multiBoxDetection(cls_score3, Loc3.data[0], common_params.num_boxes[2], min_sizes[2], max_sizes[2], common_params.steps[2], common_params.aspect_ratios[2])
    offsets4, default_boxes4, class_labels4, class_scores4 = multiBoxDetection(cls_score4, Loc4.data[0], common_params.num_boxes[3], min_sizes[3], max_sizes[3], common_params.steps[3], common_params.aspect_ratios[3])
    offsets5, default_boxes5, class_labels5, class_scores5 = multiBoxDetection(cls_score5, Loc5.data[0], common_params.num_boxes[4], min_sizes[4], max_sizes[4], common_params.steps[4], common_params.aspect_ratios[4])
    offsets6, default_boxes6, class_labels6, class_scores6 = multiBoxDetection(cls_score6, Loc6.data[0], common_params.num_boxes[5], min_sizes[5], max_sizes[5], common_params.steps[5], common_params.aspect_ratios[5])

    # オフセットベクトルによりdefault boxを補正
    candidates1 = candidatesDetection(offsets1, default_boxes1, class_labels1, class_scores1, color_img)
    candidates2 = candidatesDetection(offsets2, default_boxes2, class_labels2, class_scores2, color_img)
    candidates3 = candidatesDetection(offsets3, default_boxes3, class_labels3, class_scores3, color_img)
    candidates4 = candidatesDetection(offsets4, default_boxes4, class_labels4, class_scores4, color_img)
    candidates5 = candidatesDetection(offsets5, default_boxes5, class_labels5, class_scores5, color_img)
    candidates6 = candidatesDetection(offsets6, default_boxes6, class_labels6, class_scores6, color_img)

    # 各階層のbounding box候補を統合
    all_candidate = []
    for j in xrange(0, len(candidates1)):
        all_candidate.append(candidates1[j])
    for j in xrange(0, len(candidates2)):
        all_candidate.append(candidates2[j])
    for j in xrange(0, len(candidates3)):
        all_candidate.append(candidates3[j])
    for j in xrange(0, len(candidates4)):
        all_candidate.append(candidates4[j])
    for j in xrange(0, len(candidates5)):
        all_candidate.append(candidates5[j])
    for j in xrange(0, len(candidates6)):
        all_candidate.append(candidates6[j])

    # non-maximum suppresionによりbounding boxの最終結果を出力
    final_detections = nonMaximumSuppresion(all_candidate)
    #final_detections = all_candidate

    offset_ = 0.5
    font = cv.FONT_HERSHEY_SIMPLEX

    # Detections dimension
    # 00: xmin
    # 01: ymin
    # 02: xmax
    # 03: ymax
    # 04: class_label
    # 05: class_score


    f = open(output_dir + '/' + file_name + ".txt", 'w') # 結果書き出し用
    # 検出したbounding boxを画像に描画
    for i in xrange(0, len(final_detections)):
        class_name = labels[final_detections[i][4]]
        cls_bgr = class_color[final_detections[i][4]]
        string_space_cls = 260
        string_space_obj = 170
        p1 = int(final_detections[i][0] + offset_)
        p2 = int(final_detections[i][1] + offset_)
        p3 = int(final_detections[i][2] + offset_)
        p4 = int(final_detections[i][3] + offset_)
        final_center_x = int((p1 + p3) / 2.0)
        final_center_y = int((p2 + p4) / 2.0)
        finai_width = int(p3 - p1)
        final_height = int(p4 - p2)
        #f.write(str(final_detections[i][4]) + ' ' + str(final_center_x) + ' ' + str(final_center_y) + ' ' + str(finai_width) + ' ' + str(final_height) + ' ' + str(final_detections[i][5]) + '\n')
        f.write(str(final_detections[i][4]) + ' ' + str(p1) + ' ' + str(p2) + ' ' + str(p3) + ' ' + str(p4) + ' ' + str(final_detections[i][5]) + '\n')
        cv.rectangle(out_cls, (p1, p2), (p3, p4), (cls_bgr[0], cls_bgr[1], cls_bgr[2]), 3)
        q1 = p1
        q2 = p4
        cv.rectangle(out_cls, (q1, q2 - 25), (q1 + string_space_cls, q2), (cls_bgr[0], cls_bgr[1], cls_bgr[2]), -1)

        if final_detections[i][5] >= 1:
            cv.putText(out_cls, class_name + ": " + str(('%.2f' % final_detections[i][5])), (q1, q2 - 8), font, 0.6, (0, 0, 0), 1, cv.CV_AA)
        else:
            cv.putText(out_cls, class_name + ": " + str(('%.2f' % final_detections[i][5])), (q1, q2 - 8), font, 0.6, (255, 255, 255), 1, cv.CV_AA)

    f.close()

    # 画像の表示, 保存
    cv.imshow("Final Detections", out_cls)
    cv.imwrite(output_dir + '/' + file_name + output_type, out_cls)
    cv.waitKey(10)

    del input_data, x_data
    del Loc1, Cls1
    del Loc2, Cls2
    del Loc3, Cls3
    del Loc4, Cls4
    del Loc5, Cls5
    del Loc6, Cls6
    del cls_score1, cls_score2, cls_score3, cls_score4, cls_score5, cls_score6
    del offsets1, offsets2, offsets3, offsets4, offsets5, offsets6
    del default_boxes1, default_boxes2, default_boxes3, default_boxes4, default_boxes5, default_boxes6
    del class_labels1, class_labels2, class_labels3, class_labels4, class_labels5, class_labels6
    del class_scores1, class_scores2, class_scores3, class_scores4, class_scores5, class_scores6
    del candidates1, candidates2, candidates3, candidates4, candidates5, candidates6
    del all_candidate, final_detections
