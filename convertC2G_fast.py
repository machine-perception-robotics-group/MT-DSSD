#Convert Color img to Grayscale for Segmentation

import glob
import os
import shutil
import cv2 as cv
import numpy as np

from distutils.dir_util import copy_tree

# Original dataset dir (Source)
original_dataset = "/Volumes/External/arcdataset/"

# Dataset dir for Segmentation (Destination)
destination = original_dataset

# Image extension (e.g. png, bmp)
ext = 'png'


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
