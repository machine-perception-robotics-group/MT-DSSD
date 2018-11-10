#! /usr/bin/env python
# -*- coding: utf-8 -*-
#segmentation labelのaug paramを生成
#やってることは「セグメンテーション画像名 + '_' + Augmentation連番」をファイルに保存するだけ


from os import path
import numpy
import csv
import glob
import common_params


#テキストファイル保存
def saveTxt(filePath, data):
    f = open(filePath, 'w')
    f.write(data + "\n")
    f.close()
    print(filePath + "に保存しました")
    return 0

if __name__ == "__main__":
    #画像リスト
    inDir = path.join(common_params.images_dir, "train", "seglabel")
    #ラベル出力先
    outDir = path.join(common_params.images_dir, "train", "seglabel_aug_param")

    imageList = glob.glob(path.join(inDir, "*.png"))
    imageList.sort()

    for imagePath in imageList:
        print("Processing:" + imagePath)
        for i in range(1, common_params.augmentation_factor + 1):
            row = imagePath[imagePath.rfind('/')+1:imagePath.rfind('.')]
            saveTxt(path.join(outDir, row + '_' + str(i) + '.txt'), row + ' ')
