#! /usr/bin/env python
# -*- coding: utf-8 -*-
from glob import glob
import cv2
import numpy as np
import random
import common_params


class Augmentation:
    def __init__(self, input_img, input_seglabel, height, width):

        self.height = height
        self.width = width

        self.seg_image = np.zeros([height, width, 3])
        self.seg_label = np.zeros([height, width])

        self.seg_image = input_img
        self.seg_label = input_seglabel

    def data_trimming(self):
        """トリミング"""
        height, width, channel = self.seg_image.shape
        h_start, h_end, w_start, w_end = 0, 0, 0, 0
        input_size = (self.width, self.height)

        size_rate = random.uniform(1, 1.1)
        cut_height = int(height / size_rate)
        cut_width = int(width / size_rate)

        while True:
            h_start = random.randint(0, height - cut_height)
            h_end = h_start + cut_height
            w_start = random.randint(0, width - cut_width)
            w_end = w_start + cut_width
            if h_start < int(height / 2) < h_end:
                break

        self.seg_image = self.seg_image[h_start:h_end, w_start:w_end]
        self.seg_image = cv2.resize(self.seg_image, input_size, interpolation=cv2.INTER_LINEAR)
        self.seg_label = self.seg_label[h_start:h_end, w_start:w_end]
        self.seg_label = cv2.resize(self.seg_label, input_size, interpolation=cv2.INTER_NEAREST)

    def data_reverse(self):
        """左右反転"""
        switch_reverse = random.randint(0, 1)
        if switch_reverse == 1:
            self.seg_image = cv2.flip(self.seg_image, 1)
            self.seg_label = cv2.flip(self.seg_label, 1)

    def data_contrast(self):
        seg_image = self.seg_image
        """コントラスト調整"""
        contrast_rgb = random.randint(0, 1)
        # 3チャンネル一括
        if contrast_rgb == 1:
            min_table = random.randint(10, 100)
            max_table = random.randint(155, 245)
            diff_table = max_table - min_table

            switch_contrast = random.randint(0, 1)
            if switch_contrast == 1:
                # ハイコントラストLUT作成
                LUT_HC = np.arange(256, dtype=np.uint8)
                for i in range(0, min_table):
                    LUT_HC[i] = 0
                for i in range(min_table, max_table):
                    LUT_HC[i] = 255 * (i - min_table) / diff_table
                for i in range(max_table, 255):
                    LUT_HC[i] = 255
                self.seg_image = cv2.LUT(seg_image, LUT_HC)

            else:
                # ローコントラストLUT作成
                LUT_LC = np.arange(256, dtype=np.uint8)
                for i in range(256):
                    LUT_LC[i] = min_table + i * diff_table / 255
                self.seg_image = cv2.LUT(seg_image, LUT_LC)

        # チャンネル毎に変換
        else:
            min_table1 = random.randint(10, 50)
            max_table1 = random.randint(205, 245)
            diff_table1 = max_table1 - min_table1
            min_table2 = random.randint(10, 50)
            max_table2 = random.randint(205, 245)
            diff_table2 = max_table2 - min_table2
            min_table3 = random.randint(10, 50)
            max_table3 = random.randint(205, 245)
            diff_table3 = max_table3 - min_table3

            height, width, channel = seg_image.shape
            img1 = np.zeros([height, width], dtype=np.uint8)
            img2 = np.zeros([height, width], dtype=np.uint8)
            img3 = np.zeros([height, width], dtype=np.uint8)

            switch_contrast = random.randint(0, 1)
            if switch_contrast == 0:
                # ハイコントラストLUT作成
                LUT_HC1 = np.arange(256, dtype=np.uint8)
                LUT_HC2 = np.arange(256, dtype=np.uint8)
                LUT_HC3 = np.arange(256, dtype=np.uint8)
                for i in range(0, min_table1):
                    LUT_HC1[i] = 0
                for i in range(min_table1, max_table1):
                    LUT_HC1[i] = 255 * (i - min_table1) / diff_table1
                for i in range(max_table1, 255):
                    LUT_HC1[i] = 255

                for i in range(0, min_table2):
                    LUT_HC2[i] = 0
                for i in range(min_table2, max_table2):
                    LUT_HC2[i] = 255 * (i - min_table2) / diff_table2
                for i in range(max_table2, 255):
                    LUT_HC2[i] = 255

                for i in range(0, min_table3):
                    LUT_HC3[i] = 0
                for i in range(min_table3, max_table3):
                    LUT_HC3[i] = 255 * (i - min_table3) / diff_table3
                for i in range(max_table3, 255):
                    LUT_HC3[i] = 255

                img1[:, :] = seg_image[:, :, 0]
                img2[:, :] = seg_image[:, :, 1]
                img3[:, :] = seg_image[:, :, 2]
                img1 = cv2.LUT(img1, LUT_HC1)
                img2 = cv2.LUT(img2, LUT_HC2)
                img3 = cv2.LUT(img3, LUT_HC3)
                self.seg_image[:, :, 0] = img1[:, :]
                self.seg_image[:, :, 1] = img2[:, :]
                self.seg_image[:, :, 2] = img3[:, :]

            else:
                # ローコントラストLUT作成
                LUT_LC1 = np.arange(256, dtype=np.uint8)
                LUT_LC2 = np.arange(256, dtype=np.uint8)
                LUT_LC3 = np.arange(256, dtype=np.uint8)
                for i in range(256):
                    LUT_LC1[i] = min_table1 + i * diff_table1 / 255
                    LUT_LC2[i] = min_table2 + i * diff_table2 / 255
                    LUT_LC3[i] = min_table3 + i * diff_table3 / 255
                img1[:, :] = seg_image[:, :, 0]
                img2[:, :] = seg_image[:, :, 1]
                img3[:, :] = seg_image[:, :, 2]
                img1 = cv2.LUT(img1, LUT_LC1)
                img2 = cv2.LUT(img2, LUT_LC2)
                img3 = cv2.LUT(img3, LUT_LC3)
                self.seg_image[:, :, 0] = img1[:, :]
                self.seg_image[:, :, 1] = img2[:, :]
                self.seg_image[:, :, 2] = img3[:, :]

    def data_gamma(self):
        seg_image = self.seg_image
        """ガンマ変換"""
        gamma_switch = random.randint(0, 1)
        # 3チャンネル一括変換
        if gamma_switch == 1:
            # ガンマ変換ルックアップテーブル
            gamma = random.uniform(0.3, 4.0)
            LUT_G = np.arange(256, dtype=np.uint8)
            for i in range(256):
                LUT_G[i] = 255 * pow(float(i) / 255, 1.0 / gamma)
            self.seg_image = cv2.LUT(seg_image, LUT_G)

        # チャンネル毎に変換
        else:
            # ガンマ変換ルックアップテーブル
            gamma1 = random.uniform(0.8, 1.2)
            gamma2 = random.uniform(0.8, 1.2)
            gamma3 = random.uniform(0.8, 1.2)
            LUT_G1 = np.arange(256, dtype=np.uint8)
            LUT_G2 = np.arange(256, dtype=np.uint8)
            LUT_G3 = np.arange(256, dtype=np.uint8)

            height, width, channel = seg_image.shape
            img1 = np.zeros([height, width], dtype=np.uint8)
            img2 = np.zeros([height, width], dtype=np.uint8)
            img3 = np.zeros([height, width], dtype=np.uint8)

            for i in range(256):
                LUT_G1[i] = 255 * pow(float(i) / 255, 1.0 / gamma1)
                LUT_G2[i] = 255 * pow(float(i) / 255, 1.0 / gamma2)
                LUT_G3[i] = 255 * pow(float(i) / 255, 1.0 / gamma3)
            img1[:, :] = seg_image[:, :, 0]
            img2[:, :] = seg_image[:, :, 1]
            img3[:, :] = seg_image[:, :, 2]
            img1 = cv2.LUT(img1, LUT_G1)
            img2 = cv2.LUT(img2, LUT_G2)
            img3 = cv2.LUT(img3, LUT_G3)
            self.seg_image[:, :, 0] = img1[:, :]
            self.seg_image[:, :, 1] = img2[:, :]
            self.seg_image[:, :, 2] = img3[:, :]

    def data_noise(self):
        seg_image = self.seg_image
        height, width, channel = self.seg_image.shape

        """ノイズ付加"""
        # ガウシアンノイズ
        mean = random.randint(-15, 15)
        sigma = random.randint(1, 100)
        gauss = np.random.normal(mean, sigma, (self.height, self.width, channel))
        seg_image = seg_image + gauss
        seg_image[seg_image > 255] = 255
        seg_image[seg_image < 0] = 0
        seg_image = seg_image.astype(np.uint8)

        # saltとpepperノイズ
        s_vs_p = random.random()
        amount = random.uniform(0.0, 0.01)

        # 塩モード
        num_salt = np.ceil(amount * seg_image.size * s_vs_p)
        salt = [np.random.randint(0, i-1, int(num_salt)) for i in seg_image.shape]
        seg_image[salt[:-1]] = (255, 255, 255)

        # 胡椒モード
        num_pepper = np.ceil(amount* seg_image.size * (1. - s_vs_p))
        pepper = [np.random.randint(0, i-1, int(num_pepper)) for i in seg_image.shape]
        seg_image[pepper[:-1]] = (0, 0, 0)

        self.seg_image = seg_image

    def data_average(self):
        """平滑化"""
        filter_size1 = random.randint(2, 10)
        filter_size2 = random.randint(2, 10)
        average_filter = (filter_size1, filter_size2)
        self.seg_image = cv2.blur(self.seg_image, average_filter)

    def __call__(self):
        """ Data Augmentationの種類選択"""
        method_rand = random.random()
        if method_rand < 0.1:       # 無処理
            self.data_trimming()
            self.data_reverse()

        elif method_rand < 0.2:     # コントラスト調整
            self.data_trimming()
            self.data_reverse()
            self.data_contrast()

        elif method_rand < 0.3:     # ガンマ変換
            self.data_trimming()
            self.data_reverse()
            self.data_gamma()

        elif method_rand < 0.4:     # ノイズ付加
            self.data_trimming()
            self.data_reverse()
            self.data_noise()

        elif method_rand < 0.5:     # 平滑化
            self.data_trimming()
            self.data_reverse()
            self.data_average()

        elif method_rand < 0.6:    # コントラスト調整 + ガンマ変換
            self.data_trimming()
            self.data_reverse()
            self.data_contrast()
            self.data_gamma()

        elif method_rand < 0.7:    # コントラスト調整 + 平滑化
            self.data_trimming()
            self.data_reverse()
            self.data_contrast()
            self.data_average()

        elif method_rand < 0.8:    # コントラスト調整 + ノイズ付加
            self.data_trimming()
            self.data_reverse()
            self.data_contrast()
            self.data_noise()

        elif method_rand < 0.9:    # ガンマ変換 + ノイズ付加
            self.data_trimming()
            self.data_reverse()
            self.data_gamma()
            self.data_noise()

        else:                       # ガンマ変換 + 平滑化
            self.data_trimming()
            self.data_reverse()
            self.data_gamma()
            self.data_average()

        #チャンネル、縦、横に変換(Chainer)
        self.seg_image = (self.seg_image.swapaxes(0, 2)).swapaxes(1, 2)

    return self.seg_image, self.seg_label
