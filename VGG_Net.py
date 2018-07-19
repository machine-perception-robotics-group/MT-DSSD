#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F
import numpy as np


class VGGNet(chainer.Chain):


    def __init__(self):
        super(VGGNet, self).__init__(
            conv1_1 = L.Convolution2D(3, 64, ksize = 3, stride = 1, pad = 1),
            conv1_2 = L.Convolution2D(64, 64, ksize = 3, stride = 1, pad = 1),

            conv2_1 = L.Convolution2D(64, 128, ksize = 3, stride = 1, pad = 1),
            conv2_2 = L.Convolution2D(128, 128, ksize = 3, stride = 1, pad = 1),

            conv3_1 = L.Convolution2D(128, 256, ksize = 3, stride = 1, pad = 1),
            conv3_2 = L.Convolution2D(256, 256, ksize = 3, stride = 1, pad = 1),
            conv3_3 = L.Convolution2D(256, 256, ksize = 3, stride = 1, pad = 1),

            conv4_1 = L.Convolution2D(256, 512, ksize = 3, stride = 1, pad = 1),
            conv4_2 = L.Convolution2D(512, 512, ksize = 3, stride = 1, pad = 1),
            conv4_3 = L.Convolution2D(512, 512, ksize = 3, stride = 1, pad = 1),

            conv5_1 = L.Convolution2D(512, 512, ksize = 3, stride = 1, pad = 1),
            conv5_2 = L.Convolution2D(512, 512, ksize = 3, stride = 1, pad = 1),
            conv5_3 = L.Convolution2D(512, 512, ksize = 3, stride = 1, pad = 1),

            fc6 = L.Convolution2D(512, 1024, ksize = 3, stride = 1, pad = 1),
            fc7 = L.Convolution2D(1024, 1024, ksize = 1, stride = 1, pad = 0)
        )
        self.train = False

    def __call__(self, x):
        k1 = F.relu(self.conv1_1(x))
        k1 = F.relu(self.conv1_2(k1))

        k1 = F.max_pooling_2d(k1, ksize = 2, stride = 2, pad = 0) # pooling_1

        k1 = F.relu(self.conv2_1(k1))
        k1 = F.relu(self.conv2_2(k1))

        k1 = F.max_pooling_2d(k1, ksize = 2, stride = 2, pad = 0) # pooling_2

        k1 = F.relu(self.conv3_1(k1))
        k1 = F.relu(self.conv3_2(k1))
        k1 = F.relu(self.conv3_3(k1))

        k1 = F.max_pooling_2d(k1, ksize = 2, stride = 2, pad = 0) # pooling_3

        k1 = F.relu(self.conv4_1(k1))
        k1 = F.relu(self.conv4_2(k1))
        k1 = F.relu(self.conv4_3(k1))

        k2 = F.max_pooling_2d(k1, ksize = 2, stride = 2, pad = 0) # pooling_4

        k2 = F.relu(self.conv5_1(k2))
        k2 = F.relu(self.conv5_2(k2))
        k2 = F.relu(self.conv5_3(k2))

        k2 = F.max_pooling_2d(k2, ksize = 3, stride = 1, pad = 1) # pooling_5

        k2 = F.relu(self.fc6(k2))
        k2 = F.relu(self.fc7(k2))


        return (k1, k2)

