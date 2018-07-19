#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from glob import glob
import cv2
import random

class Dataloader:

    def Train(self, rgb_path, label_path, batch, height, width):
        IMAGE_FILE = glob(rgb_path)
        LABEL_FILE = glob(label_path)

        IMAGE_FILE.sort()
        LABEL_FILE.sort()

        total_image = len(IMAGE_FILE)
        size_image = (width, height)

        image = np.zeros((batch, height, width, 3)).astype(np.float32)
        label = np.zeros((batch, height, width)).astype(np.int32)

        for counter in xrange(batch):
            choice = random.randint(0, total_image-1)

            img = cv2.imread(IMAGE_FILE[choice], 1)
            resize = cv2.resize(img, size_image, interpolation=cv2.INTER_LINEAR)
            image[counter, :, :, :] = resize

            img = cv2.imread(LABEL_FILE[choice], 0)
            resize = cv2.resize(img, size_image, interpolation=cv2.INTER_NEAREST)
            label[counter, :, :] = resize

        image = image.swapaxes(1, 3)
        image = image.swapaxes(2, 3)

        return image, label


    def Test(self, rgb_path, label_path, num, height, width):
        IMAGE_FILE = glob(rgb_path)
        LABEL_FILE = glob(label_path)

        IMAGE_FILE.sort()
        LABEL_FILE.sort()

        size_image = (width, height)

        image = np.zeros((1, height, width, 3)).astype(np.float32)
        label = np.zeros((1, height, width)).astype(np.int32)

        img = cv2.imread(IMAGE_FILE[num], 1)
        resize = cv2.resize(img, size_image, interpolation=cv2.INTER_LINEAR)
        image[0, :, :, :] = resize

        img = cv2.imread(LABEL_FILE[num], 0)
        resize = cv2.resize(img, size_image, interpolation=cv2.INTER_NEAREST)
        label[0, :, :] = resize

        image = image.swapaxes(1, 3)
        image = image.swapaxes(2, 3)

        return image, label
