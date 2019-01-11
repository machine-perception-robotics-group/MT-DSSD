#! /usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F
import numpy as np
import common_params
from chainer import cuda
import matplotlib.pyplot as plt


class SSDNet(chainer.Chain):

    def __init__(self):
        super(SSDNet, self).__init__(

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
            fc7 = L.Convolution2D(1024, 1024, ksize = 1, stride = 1, pad = 0),

            conv6_1 = L.Convolution2D(1024, 256, ksize = 1, stride = 1, pad = 0),
            conv6_2 = L.Convolution2D(256, 512, ksize = 3, stride = 2, pad = 1),
            bn_conv6 = L.BatchNormalization(512),

            conv7_1 = L.Convolution2D(512, 128, ksize = 1, stride = 1, pad = 0),
            conv7_2 = L.Convolution2D(128, 256, ksize = 3, stride = 2, pad = 1),

            conv8_1 = L.Convolution2D(256, 128, ksize = 1, stride = 1, pad = 0),
            conv8_2 = L.Convolution2D(128, 256, ksize = 3, stride = 1, pad = 0),

            conv9_1 = L.Convolution2D(256, 128, ksize = 1, stride = 1, pad = 0),
            conv9_2 = L.Convolution2D(128, 256, ksize = 3, stride = 1, pad = 0),

            conv9_2_mbox_loc = L.Convolution2D(256, common_params.num_of_offset_dims * common_params.num_boxes[5], ksize = 3, stride = 1, pad = 1),
            conv9_2_mbox_cls = L.Convolution2D(256, common_params.num_of_classes * common_params.num_boxes[5], ksize = 3, stride = 1, pad = 1),

            deconv9_2_1 = L.Deconvolution2D(256, common_params.num_of_classes, ksize = 10, stride = 1, pad = 0), #10×10
            deconv9_2_2 = L.Deconvolution2D(common_params.num_of_classes, common_params.num_of_classes, ksize = 11, stride = 8, pad = 4), #75×75
            deconv9_2_3 = L.Deconvolution2D(common_params.num_of_classes, common_params.num_of_classes, ksize = 6, stride = 4, pad = 1), #300×300

            deconv1 = L.Deconvolution2D(256, 256, ksize = 3, stride = 1, pad = 0), #3×3

            conv8_2_mbox_loc = L.Convolution2D(256, common_params.num_of_offset_dims * common_params.num_boxes[4], ksize = 3, stride = 1, pad = 1),
            conv8_2_mbox_cls = L.Convolution2D(256, common_params.num_of_classes * common_params.num_boxes[4], ksize = 3, stride = 1, pad = 1),

            deconv8_2_1 = L.Deconvolution2D(256, common_params.num_of_classes, ksize = 9, stride = 6, pad = 1), #19×19
            deconv8_2_2 = L.Deconvolution2D(common_params.num_of_classes, common_params.num_of_classes, ksize = 7, stride = 4, pad = 2), #75×75
            deconv8_2_3 = L.Deconvolution2D(common_params.num_of_classes, common_params.num_of_classes, ksize = 6, stride = 4, pad = 1), #300×300

            deconv2 = L.Deconvolution2D(256, 256, ksize = 3, stride = 2, pad = 1), #5×5

            conv7_2_mbox_loc = L.Convolution2D(256, common_params.num_of_offset_dims * common_params.num_boxes[3], ksize = 3, stride = 1, pad = 1),
            conv7_2_mbox_cls = L.Convolution2D(256, common_params.num_of_classes * common_params.num_boxes[3], ksize = 3, stride = 1, pad = 1),

            deconv7_2_1 = L.Deconvolution2D(256, common_params.num_of_classes, ksize = 7, stride = 4, pad = 2), #19×19
            deconv7_2_2 = L.Deconvolution2D(common_params.num_of_classes, common_params.num_of_classes, ksize = 7, stride = 4, pad = 2), #75×75
            deconv7_2_3 = L.Deconvolution2D(common_params.num_of_classes, common_params.num_of_classes, ksize = 6, stride = 4, pad = 1), #300×300

            deconv3 = L.Deconvolution2D(256, 512, ksize = 4, stride = 2, pad = 1), #10×10

            conv6_2_mbox_loc = L.Convolution2D(512, common_params.num_of_offset_dims * common_params.num_boxes[2], ksize = 3, stride = 1, pad = 1),
            conv6_2_mbox_cls = L.Convolution2D(512, common_params.num_of_classes * common_params.num_boxes[2], ksize = 3, stride = 1, pad = 1),

            deconv6_2_1 = L.Deconvolution2D(512, common_params.num_of_classes, ksize = 6, stride = 4, pad = 2), #38×38
            deconv6_2_2 = L.Deconvolution2D(common_params.num_of_classes, common_params.num_of_classes, ksize = 3, stride = 2, pad = 1), #75×75
            deconv6_2_3 = L.Deconvolution2D(common_params.num_of_classes, common_params.num_of_classes, ksize = 6, stride = 4, pad = 1), #300×300


            deconv4 = L.Deconvolution2D(512, 1024, ksize = 3, stride = 2, pad = 1), #19×19

            fc7_mbox_loc = L.Convolution2D(1024, common_params.num_of_offset_dims * common_params.num_boxes[1], ksize = 3, stride = 1, pad = 1),
            fc7_mbox_cls = L.Convolution2D(1024, common_params.num_of_classes * common_params.num_boxes[1], ksize = 3, stride = 1, pad = 1),

            deconvfc7_1 = L.Deconvolution2D(1024, common_params.num_of_classes, ksize = 4, stride = 2, pad = 1), #38×38
            deconvfc7_2 = L.Deconvolution2D(common_params.num_of_classes, common_params.num_of_classes, ksize = 3, stride = 2, pad = 1), #75×75
            deconvfc7_3 = L.Deconvolution2D(common_params.num_of_classes, common_params.num_of_classes, ksize = 6, stride = 4, pad = 1), #300×300

            deconv5 = L.Deconvolution2D(1024, 512, ksize = 4, stride = 2, pad = 1), #38×38

            bn4_3 = L.BatchNormalization(512),

            conv4_3_norm_mbox_loc = L.Convolution2D(512, common_params.num_of_offset_dims * common_params.num_boxes[0], ksize = 3, stride = 1, pad = 1),
            conv4_3_norm_mbox_cls = L.Convolution2D(512, common_params.num_of_classes * common_params.num_boxes[0], ksize = 3, stride = 1, pad = 1),

            deconv4_3_1 = L.Deconvolution2D(512, common_params.num_of_classes, ksize = 3, stride = 2, pad = 1), #75×75
            deconv4_3_2 = L.Deconvolution2D(common_params.num_of_classes, common_params.num_of_classes, ksize = 4, stride = 2, pad = 1), #150×150
            deconv4_3_3 = L.Deconvolution2D(common_params.num_of_classes, common_params.num_of_classes, ksize = 4, stride = 2, pad = 1),  #300×300

            segconv = L.Convolution2D(common_params.num_of_classes*6, common_params.num_of_classes, ksize = 1, stride = 1, pad = 0)
        )
        self.train = False

    def __call__(self, x):

        k1 = F.relu(self.conv1_1(x))
        k1 = self.conv1_2(k1)
        k1 = F.relu(k1)

        k1 = F.max_pooling_2d(k1, ksize = 2, stride = 2, pad = 0) # pooling_1

        k1 = F.relu(self.conv2_1(k1))
        k1 = self.conv2_2(k1)
        k1 = F.relu(k1)

        k1 = F.max_pooling_2d(k1, ksize = 2, stride = 2, pad = 0) # pooling_2

        k1 = F.relu(self.conv3_1(k1))
        k1 = F.relu(self.conv3_2(k1))
        k1 = self.conv3_3(k1)
        k1 = F.relu(k1)

        k1 = F.max_pooling_2d(k1, ksize = 2, stride = 2, pad = 0) # pooling_3

        k1 = F.relu(self.conv4_1(k1))
        k1 = F.relu(self.conv4_2(k1))
        k1 = self.conv4_3(k1)
        k1 = F.relu(k1)

        k2 = F.max_pooling_2d(k1, ksize = 2, stride = 2, pad = 0) # pooling_4

        k2 = F.relu(self.conv5_1(k2))
        k2 = F.relu(self.conv5_2(k2))
        k2 = self.conv5_3(k2)
        k2 = F.relu(k2)

        k2 = F.max_pooling_2d(k2, ksize = 3, stride = 1, pad = 1) # pooling_5

        k2 = F.relu(self.fc6(k2))
        k2 = self.fc7(k2)
        k2 = F.relu(k2)

        k3 = F.relu(self.conv6_1(k2))
        k3 = self.conv6_2(k3)
        k3 = F.relu(k3)

        k4 = F.relu(self.conv7_1(k3))
        k4 = self.conv7_2(k4)
        k4 = F.relu(k4)

        k5 = F.relu(self.conv8_1(k4))
        k5 = self.conv8_2(k5)
        k5 = F.relu(k5)

        k6 = F.relu(self.conv9_1(k5))
        k6 = self.conv9_2(k6)
        k6 = F.relu(k6)

        Loc6 = self.conv9_2_mbox_loc(k6)  # Box_Estimator_6
        Cls6 = self.conv9_2_mbox_cls(k6) # Class_Classifier_6

        Seg1 = self.deconv9_2_1(k6)
        Seg1 = self.deconv9_2_2(Seg1)
        Seg1 = self.deconv9_2_3(Seg1)

        y5 = self.deconv1(k6) #3×3

        yy5 = y5 + k5
        yy5 = F.relu(yy5)

        Loc5 = self.conv8_2_mbox_loc(yy5)  # Box_Estimator_5
        Cls5 = self.conv8_2_mbox_cls(yy5) # Class_Classifier_5

        Seg2 = self.deconv8_2_1(yy5)
        Seg2 = self.deconv8_2_2(Seg2)
        Seg2 = self.deconv8_2_3(Seg2)

        y4 = self.deconv2(yy5)

        yy4 = y4 + k4
        yy4 = F.relu(yy4)

        Loc4 = self.conv7_2_mbox_loc(yy4)  # Box_Estimator_4
        Cls4 = self.conv7_2_mbox_cls(yy4) # Class_Classifier_4

        Seg3 = self.deconv7_2_1(yy4)
        Seg3 = self.deconv7_2_2(Seg3)
        Seg3 = self.deconv7_2_3(Seg3)

        y3 = self.deconv3(yy4)

        yy3 = y3 + k3
        yy3 = F.relu(yy3)

        Loc3 = self.conv6_2_mbox_loc(yy3)  # Box_Estimator_3
        Cls3 = self.conv6_2_mbox_cls(yy3) # Class_Classifier_3

        Seg4 = self.deconv6_2_1(yy3)
        Seg4 = self.deconv6_2_2(Seg4)
        Seg4 = self.deconv6_2_3(Seg4)

        y2 = self.deconv4(yy3)

        yy2 = y2 + k2
        yy2 = F.relu(yy2)

        Loc2 = self.fc7_mbox_loc(yy2)  # Box_Estimator_2
        Cls2 = self.fc7_mbox_cls(yy2) # Class_Classifier_2

        Seg5 = self.deconvfc7_1(yy2)
        Seg5 = self.deconvfc7_2(Seg5)
        Seg5 = self.deconvfc7_3(Seg5)

        y1 = self.deconv5(yy2)

        yy1 = y1 + k1
        yy1 = F.relu(yy1)

        Loc1 = self.conv4_3_norm_mbox_loc(self.bn4_3(yy1))  # Box_Estimator_1
        Cls1 = self.conv4_3_norm_mbox_cls(self.bn4_3(yy1)) # Class_Classifier_1

        Seg6 = self.deconv4_3_1(yy1) #75×75
        Seg6 = self.deconv4_3_2(Seg6) #150×150
        Seg6 = self.deconv4_3_3(Seg6) #300×300

        Seg = F.concat([Seg1, Seg2, Seg3, Seg4, Seg5, Seg6], axis = 1)
        #Seg = Seg1 + Seg2 + Seg3 + Seg4 + Seg5 + Seg6

        Seg = self.segconv(Seg)

        #特徴マップ可視化
        #xlabel = np.array(range(38))
        #ylabel = np.array(range(38))
        #fig = cuda.to_cpu(Seg6.data)
        #for i in range(0, len(fig[:, 0, 0, 0])):
        #    for j in range(0, len(fig[0, :, 0, 0])):
        #        a = fig[i, j]

        #fig, ax = plt.subplots()
        #heatmap = ax.pcolor(a, cmap = plt.cm.Blues)

        #ax.set_xticks(np.arange(a.shape[0]) + 0.5, minor=False)
        #ax.set_yticks(np.arange(a.shape[1]) + 0.5, minor=False)

        #ax.invert_yaxis()
        #ax.xaxis.tick_top()

        #ax.set_xticklabels(xlabel, minor=False)
        #ax.set_yticklabels(ylabel, minor=False)
        #plt.show()
        #plt.savefig('image.png')

        if self.train:
            Loc1 = F.transpose(Loc1, [0, 2, 3, 1]) #(バッチ数,高さ,幅,チャンネル数)
            Cls1 = F.transpose(Cls1, [0, 2, 3, 1])

            Loc2 = F.transpose(Loc2, [0, 2, 3, 1])
            Cls2 = F.transpose(Cls2, [0, 2, 3, 1])

            Loc3 = F.transpose(Loc3, [0, 2, 3, 1])
            Cls3 = F.transpose(Cls3, [0, 2, 3, 1])

            Loc4 = F.transpose(Loc4, [0, 2, 3, 1])
            Cls4 = F.transpose(Cls4, [0, 2, 3, 1])

            Loc5 = F.transpose(Loc5, [0, 2, 3, 1])
            Cls5 = F.transpose(Cls5, [0, 2, 3, 1])

            Loc6 = F.transpose(Loc6, [0, 2, 3, 1])
            Cls6 = F.transpose(Cls6, [0, 2, 3, 1])

            Loc1 = F.reshape(Loc1, [Loc1.data.shape[0] * Loc1.data.shape[1] * Loc1.data.shape[2] * common_params.num_boxes[0], int(Loc1.data.shape[3] / common_params.num_boxes[0])])
            Cls1 = F.reshape(Cls1, [Cls1.data.shape[0] * Cls1.data.shape[1] * Cls1.data.shape[2] * common_params.num_boxes[0], int(Cls1.data.shape[3] / common_params.num_boxes[0])])

            Loc2 = F.reshape(Loc2, [Loc2.data.shape[0] * Loc2.data.shape[1] * Loc2.data.shape[2] * common_params.num_boxes[1], int(Loc2.data.shape[3] / common_params.num_boxes[1])])
            Cls2 = F.reshape(Cls2, [Cls2.data.shape[0] * Cls2.data.shape[1] * Cls2.data.shape[2] * common_params.num_boxes[1], int(Cls2.data.shape[3] / common_params.num_boxes[1])])

            Loc3 = F.reshape(Loc3, [Loc3.data.shape[0] * Loc3.data.shape[1] * Loc3.data.shape[2] * common_params.num_boxes[2], int(Loc3.data.shape[3] / common_params.num_boxes[2])])
            Cls3 = F.reshape(Cls3, [Cls3.data.shape[0] * Cls3.data.shape[1] * Cls3.data.shape[2] * common_params.num_boxes[2], int(Cls3.data.shape[3] / common_params.num_boxes[2])])

            Loc4 = F.reshape(Loc4, [Loc4.data.shape[0] * Loc4.data.shape[1] * Loc4.data.shape[2] * common_params.num_boxes[3], int(Loc4.data.shape[3] / common_params.num_boxes[3])])
            Cls4 = F.reshape(Cls4, [Cls4.data.shape[0] * Cls4.data.shape[1] * Cls4.data.shape[2] * common_params.num_boxes[3], int(Cls4.data.shape[3] / common_params.num_boxes[3])])

            Loc5 = F.reshape(Loc5, [Loc5.data.shape[0] * Loc5.data.shape[1] * Loc5.data.shape[2] * common_params.num_boxes[4], int(Loc5.data.shape[3] / common_params.num_boxes[4])])
            Cls5 = F.reshape(Cls5, [Cls5.data.shape[0] * Cls5.data.shape[1] * Cls5.data.shape[2] * common_params.num_boxes[4], int(Cls5.data.shape[3] / common_params.num_boxes[4])])

            Loc6 = F.reshape(Loc6, [Loc6.data.shape[0] * Loc6.data.shape[1] * Loc6.data.shape[2] * common_params.num_boxes[5], int(Loc6.data.shape[3] / common_params.num_boxes[5])])
            Cls6 = F.reshape(Cls6, [Cls6.data.shape[0] * Cls6.data.shape[1] * Cls6.data.shape[2] * common_params.num_boxes[5], int(Cls6.data.shape[3] / common_params.num_boxes[5])])
            return (Loc1, Cls1, Loc2, Cls2, Loc3, Cls3, Loc4, Cls4, Loc5, Cls5, Loc6, Cls6, Seg)

        else:
            return (Loc1, Cls1, Loc2, Cls2, Loc3, Cls3, Loc4, Cls4, Loc5, Cls5, Loc6, Cls6, Seg)
