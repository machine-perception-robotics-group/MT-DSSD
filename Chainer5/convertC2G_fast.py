#Convert Color img to Grayscale for Segmentation

import glob
import os
import shutil
import cv2 as cv
import numpy as np

import common_params

from distutils.dir_util import copy_tree

# Original dataset dir (Source)
original_dataset = common_params.images_dir

# Dataset dir for Segmentation (Destination)
destination = original_dataset

# Image extension (e.g. png, bmp)
ext = 'png'

class_color = common_params.arc_class_color[:, ::-1]


def cvResize(src, w, h, inter):
    img = cv.imread(src)
    if img is None :
        print( "[ERROR]Cannot read image: " + src )
        sys.exit()
    img = cv.resize(img, (w, h), interpolation=inter)
    return img


def convertGray(img):
    for i in range(0, class_color.shape[0]):
        img[(img == class_color[i]).all(axis=2)] = [i, i, i]
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img


def copyResize(src, dest, type):
    src_list = glob.glob(os.path.join(src, '*.' + ext))

    for src_path in src_list:
        print(src_path)
        if type == "color":
            img = cvResize(src_path, 480, 360, cv.INTER_CUBIC)
        elif type == "gray":
            img = cvResize(src_path, 480, 360, cv.INTER_NEAREST)
            img = convertGray(img)
        cv.imwrite(os.path.join(dest, os.path.basename(src_path)), img)



def main():
    # Source dir
    src_trainannot_path = os.path.join(original_dataset, "train", "segmentation")

    # Destination dir init
    dest_trainannot_path = os.path.join(destination, "train", "seglabel")
    os.mkdir(dest_trainannot_path)

    copyResize(src_trainannot_path, dest_trainannot_path, "gray")

    print("[Info] Done.")

if __name__ == '__main__':
    main()
