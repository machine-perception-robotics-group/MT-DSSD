#coding: utf-8
from glob import glob
from os import path
import os
import cv2 as cv
import sys
import math
import numpy as np

import common_params
from data_augmentation import augmentation

# クラスラベル (クラス名にはスペース(空白)は禁止)
arc_labels = ["Background",
              "Binder",
              "Balloons",
              "Baby_Wipes",
              "Toilet_Brush",
              "Toothbrushes",
              "Crayons",
              "Salts",
              "DVD",
              "Glue_Sticks",
              "Eraser",
              "Scissors",
              "Green_Book",
              "Socks",
              "Irish_Spring",
              "Paper_Tape",
              "Touch_Tissues",
              "Knit_Gloves",
              "Laugh_Out_Loud_Jokes",
              "Pencil_Cup",
              "Mini_Marbles",
              "Neoprene_Weight",
              "Wine_Glasses",
              "Water_Bottle",
              "Reynolds_Pie",
              "Reynolds_Wrap",
              "Robots_Everywhere",
              "Duct_Tape",
              "Sponges",
              "Speed_Stick",
              "Index_Cards",
              "Ice_Cube_Tray",
              "Table_Cover",
              "Measuring_Spoons",
              "Bath_Sponge",
              "Pencils",
              "Mousetraps",
              "Face_Cloth",
              "Tennis_Balls",
              "Spray_Bottle",
              "Flashlights"]

labels = arc_labels

# Ground truth boxとDefault boxの重なり率を計算
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


# 重なり率の閾値 (閾値以上の重なり率のDefault boxはpositiveとする)
ovarlap_th = 0.5


step = int(math.floor((common_params.max_ratio - common_params.min_ratio) / (len(common_params.mbox_source_layers) - 2)))

min_sizes = []
max_sizes = []

# Default boxの最小・最大サイズを計算
for ratio in xrange(common_params.min_ratio, common_params.max_ratio + 1, step):
    min_sizes.append(common_params.insize * ratio / 100.)
    max_sizes.append(common_params.insize * (ratio + step) / 100.)

min_sizes = [common_params.insize * 10 / 100.] + min_sizes
max_sizes = [common_params.insize * 20 / 100.] + max_sizes

# アノテーション(Ground truth boxの教示)ファイルのパス
lists = open('./img_name_list.txt', 'r')
b = 1
# アノテーションファイルのループ
for fl in lists:

    print fl
    print b
    b += 1

    fname = fl[0:-1]

    # アノテーションファイルをオープン
    f = open(common_params.images_dir + '/train/boundingbox/' + fname + '.txt')

    # クラスidの配列
    original_idx = []

    # Ground truth boxの配列
    original_gt_boxes = []

    # Ground truth boxの左上座標と右下座標、boxのクラスidを格納
    for rw in f:
        ln = rw.split(' ')
        original_idx.append(int(ln[0]))
        xmin = (float(ln[1]) - float(ln[3]) / 2.) * common_params.insize
        ymin = (float(ln[2]) - float(ln[4]) / 2.) * common_params.insize
        xmax = (float(ln[1]) + float(ln[3]) / 2.) * common_params.insize
        ymax = (float(ln[2]) + float(ln[4]) / 2.) * common_params.insize
        original_gt_boxes.append([xmin, ymin, xmax, ymax])

    f.close()

    # アノテーションファイルに対応する画像を読み込み
    original_img = cv.imread(common_params.images_dir + '/train/rgb/' + fname + '.png', cv.IMREAD_COLOR)

    if original_img is None:
        print('Input image error')
        print(common_params.images_dir + '/train/rgb/' + fname + '.png')
        sys.exit(1)

    # 画像をSSDの入力サイズにリサイズ
    original_img = cv.resize(original_img, (common_params.insize, common_params.insize), interpolation = cv.INTER_CUBIC)

    for ag in xrange(0, common_params.augmentation_factor):

        input_img = original_img.copy()
        idx = original_idx #クラス
        gt_boxes = original_gt_boxes

        if (ag >= 1):
            input_img, idx, gt_boxes, border_pixels, crop_param, hsv_param, flip_type = augmentation(input_img, idx, gt_boxes)

        # ---Augmentation画像の確認が不要な場合は以下9行をコメントアウト------------
        out2 = input_img.copy()
        for bx in xrange(0, len(gt_boxes)):
            p1 = int(gt_boxes[bx][0])
            p2 = int(gt_boxes[bx][1])
            p3 = int(gt_boxes[bx][2])
            p4 = int(gt_boxes[bx][3])
            cv.rectangle(out2, (p1, p2), (p3, p4), (0, 255, 0), 2)

        cv.imshow('Augmentation', out2)
        cv.waitKey(10)
        # ----------------------------------------------------------------

        img_width = input_img.shape[1]
        img_height = input_img.shape[0]

        # positive教示データとnegative教示データ
        fout_pos = open(common_params.images_dir + '/train/positives/' + fname + '_' + str(ag) + '.txt', 'w')
        fout_neg = open(common_params.images_dir + '/train/negatives/' + fname + '_' + str(ag) + '.txt', 'w')

        fout_aug = open(common_params.images_dir + '/train/img_aug_param/' + fname + '_' + str(ag) + '.txt', 'w')

        out_line = '{} \n'.format(fname)
        fout_aug.write(out_line)

        if (ag >= 1):
            out_line = '{} \n'.format(1)
            fout_aug.write(out_line)
            out_line = '{} {} {} {} \n'.format(border_pixels[0], border_pixels[1], border_pixels[2], border_pixels[3])
            fout_aug.write(out_line)
            out_line = '{} {} {} {} \n'.format(crop_param[0], crop_param[1], crop_param[2], crop_param[3])
            fout_aug.write(out_line)
            out_line = '{} {} {} \n'.format(hsv_param[0], hsv_param[1], hsv_param[2])
            fout_aug.write(out_line)
            out_line = '{} \n'.format(flip_type)
            fout_aug.write(out_line)
        else:
            out_line = '{} \n'.format(0)
            fout_aug.write(out_line)
            out_line = '{} {} {} {} \n'.format(999, 999, 999, 999)
            fout_aug.write(out_line)
            out_line = '{} {} {} {} \n'.format(999, 999, 999, 999)
            fout_aug.write(out_line)
            out_line = '{} {} {} \n'.format(999., 999., 999.)
            fout_aug.write(out_line)
            out_line = '{} \n'.format(999)
            fout_aug.write(out_line)

        # 特徴マップのループ
        for j in xrange(0, len(common_params.map_sizes)):

            # 特徴マップのピクセル数
            map_dim = common_params.map_sizes[j] * common_params.map_sizes[j]

            # 特徴マップの位置のループ
            for k in xrange(0, map_dim):

                # 特徴マップの位置(x, y)
                c = int(k % common_params.map_sizes[j])
                r = int(k / common_params.map_sizes[j])

                center_x = float((c + 0.5) * common_params.steps[j])
                center_y = float((r + 0.5) * common_params.steps[j])

                # DF boxのループ
                for l in xrange(0, common_params.num_boxes[j]):

                    # 1〜6番目のDefault boxの右上座標と左上座標を計算
                    if l == 0:
                            box_width = box_height = min_sizes[j]
                            xmin = (center_x - box_width / 2.) * (1. / common_params.insize)
                            ymin = (center_y - box_height / 2.) * (1./ common_params.insize)
                            xmax = (center_x + box_width / 2.) * (1. / common_params.insize)
                            ymax = (center_y + box_height / 2.) * (1. / common_params.insize)
                    elif l == 1:
                            box_width = box_height = np.sqrt(min_sizes[j] * max_sizes[j])
                            xmin = (center_x - box_width / 2.) * (1. / common_params.insize)
                            ymin = (center_y - box_height / 2.) * (1. / common_params.insize)
                            xmax = (center_x + box_width / 2.) * (1. / common_params.insize)
                            ymax = (center_y + box_height / 2.) * (1. / common_params.insize)
                    elif l == 2:
                            box_width = min_sizes[j] * np.sqrt(float(common_params.aspect_ratios[j][0]))
                            box_height = min_sizes[j] / np.sqrt(float(common_params.aspect_ratios[j][0]))
                            xmin = (center_x - box_width / 2.) * (1. / common_params.insize)
                            ymin = (center_y - box_height / 2.) * (1. / common_params.insize)
                            xmax = (center_x + box_width / 2.) * (1. / common_params.insize)
                            ymax = (center_y + box_height / 2.) * (1. / common_params.insize)
                    elif l == 3:
                            box_width = min_sizes[j] * np.sqrt(1. / float(common_params.aspect_ratios[j][0]))
                            box_height = min_sizes[j] / np.sqrt(1. / float(common_params.aspect_ratios[j][0]))
                            xmin = (center_x - box_width / 2.) * (1. / common_params.insize)
                            ymin = (center_y - box_height / 2.) * (1. / common_params.insize)
                            xmax = (center_x + box_width / 2.) * (1. / common_params.insize)
                            ymax = (center_y + box_height / 2.) * (1. / common_params.insize)
                    elif l == 4:
                            box_width = min_sizes[j] * np.sqrt(float(common_params.aspect_ratios[j][1]))
                            box_height = min_sizes[j] / np.sqrt(float(common_params.aspect_ratios[j][1]))
                            xmin = (center_x - box_width / 2.) * (1. / common_params.insize)
                            ymin = (center_y - box_height / 2.) * (1. / common_params.insize)
                            xmax = (center_x + box_width / 2.) * (1. / common_params.insize)
                            ymax = (center_y + box_height / 2.) * (1. / common_params.insize)
                    elif l == 5:
                            box_width = min_sizes[j] * np.sqrt(1. / float(common_params.aspect_ratios[j][1]))
                            box_height = min_sizes[j] / np.sqrt(1. / float(common_params.aspect_ratios[j][1]))
                            xmin = (center_x - box_width / 2.) * (1. / common_params.insize)
                            ymin = (center_y - box_height / 2.) * (1. / common_params.insize)
                            xmax = (center_x + box_width / 2.) * (1. / common_params.insize)
                            ymax = (center_y + box_height / 2.) * (1. / common_params.insize)


                    # Default boxの座標範囲を入力画像サイズに広げる
                    xmin = min(max(xmin, 0.), 1.) * common_params.insize
                    ymin = min(max(ymin, 0.), 1.) * common_params.insize
                    xmax = min(max(xmax, 0.), 1.) * common_params.insize
                    ymax = min(max(ymax, 0.), 1.) * common_params.insize

                    df_box = [xmin, ymin, xmax, ymax]

                    # GT boxのループ
                    for i in xrange(0, len(gt_boxes)):

                        # Default boxとGround truth boxの重なり率を計算
                        overlap = jaccardOverlap(gt_boxes[i], df_box)

                        gt_xmin = gt_boxes[i][0] / common_params.insize
                        gt_ymin = gt_boxes[i][1] / common_params.insize
                        gt_xmax = gt_boxes[i][2] / common_params.insize
                        gt_ymax = gt_boxes[i][3] / common_params.insize

                        df_xmin = df_box[0] / common_params.insize
                        df_ymin = df_box[1] / common_params.insize
                        df_xmax = df_box[2] / common_params.insize
                        df_ymax = df_box[3] / common_params.insize

                        # 教示データの書き出す内容
                        # クラスの名前, クラスID, GT左上x座標, GT左上y座標, GT右下x座標, GT右下y座標, DF左上x座標, DF左上y座標, DF右下x座標, DF右下y座標, GT boxのindex, feature mapの階層のindex, feature mapの座標のindex, DF boxのindex, 重なり率

                        if overlap >= ovarlap_th:
                            # 重なり率が閾値以上の場合はpositive sample
                            cls_name = labels[idx[i]]
                            cls_id = idx[i]
                            out_line = '{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} \n'.format(cls_name, cls_id, gt_xmin, gt_ymin, gt_xmax, gt_ymax, df_xmin, df_ymin, df_xmax, df_ymax, i, j, k, l, overlap)
                            fout_pos.write(out_line)
                        elif overlap >= 0.1:
                            # 重なり率が閾値に満たさないが0.1以上の場合はhard negative sample
                            cls_name = labels[0]
                            cls_id = 0
                            out_line = '{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} \n'.format(cls_name, cls_id, gt_xmin, gt_ymin, gt_xmax, gt_ymax, df_xmin, df_ymin, df_xmax, df_ymax, i, j, k, l, overlap)
                            fout_neg.write(out_line)
