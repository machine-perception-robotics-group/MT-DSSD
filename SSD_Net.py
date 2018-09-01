#! /usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F
import numpy as np
import common_params


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

            bn4_3 = L.BatchNormalization(512),

            conv4_3_mbox_loc = L.Convolution2D(512, common_params.num_of_offset_dims * common_params.num_boxes[0], ksize = 3, stride = 1, pad = 1),
            conv4_3_mbox_cls = L.Convolution2D(512, common_params.num_of_classes * common_params.num_boxes[0], ksize = 3, stride = 1, pad = 1),

            conv5_1 = L.Convolution2D(512, 512, ksize = 3, stride = 1, pad = 1),
            conv5_2 = L.Convolution2D(512, 512, ksize = 3, stride = 1, pad = 1),
            conv5_3 = L.Convolution2D(512, 512, ksize = 3, stride = 1, pad = 1),

            fc6 = L.Convolution2D(512, 1024, ksize = 3, stride = 1, pad = 1),
            fc7 = L.Convolution2D(1024, 1024, ksize = 1, stride = 1, pad = 0),

            #bn7 = L.BatchNormalization(1024),

            fc7_mbox_loc = L.Convolution2D(1024, common_params.num_of_offset_dims * common_params.num_boxes[1], ksize = 3, stride = 1, pad = 1),
            fc7_mbox_cls = L.Convolution2D(1024, common_params.num_of_classes * common_params.num_boxes[1], ksize = 3, stride = 1, pad = 1),

            conv6_1 = L.Convolution2D(1024, 256, ksize = 1, stride = 1, pad = 0),
            conv6_2 = L.Convolution2D(256, 512, ksize = 3, stride = 2, pad = 1),

            #bn6_2 = L.BatchNormalization(512),

            conv6_2_mbox_loc = L.Convolution2D(512, common_params.num_of_offset_dims * common_params.num_boxes[2], ksize = 3, stride = 1, pad = 1),
            conv6_2_mbox_cls = L.Convolution2D(512, common_params.num_of_classes * common_params.num_boxes[2], ksize = 3, stride = 1, pad = 1),

            conv7_1 = L.Convolution2D(512, 128, ksize = 1, stride = 1, pad = 0),
            conv7_2 = L.Convolution2D(128, 256, ksize = 3, stride = 2, pad = 1),

            #bn7_2 = L.BatchNormalization(256),

            conv7_2_mbox_loc = L.Convolution2D(256, common_params.num_of_offset_dims * common_params.num_boxes[3], ksize = 3, stride = 1, pad = 1),
            conv7_2_mbox_cls = L.Convolution2D(256, common_params.num_of_classes * common_params.num_boxes[3], ksize = 3, stride = 1, pad = 1),

            conv8_1 = L.Convolution2D(256, 128, ksize = 1, stride = 1, pad = 0),
            conv8_2 = L.Convolution2D(128, 256, ksize = 3, stride = 1, pad = 0),

            #bn8_2 = L.BatchNormalization(256),

            conv8_2_mbox_loc = L.Convolution2D(256, common_params.num_of_offset_dims * common_params.num_boxes[4], ksize = 3, stride = 1, pad = 1),
            conv8_2_mbox_cls = L.Convolution2D(256, common_params.num_of_classes * common_params.num_boxes[4], ksize = 3, stride = 1, pad = 1),

            conv9_1 = L.Convolution2D(256, 128, ksize = 1, stride = 1, pad = 0),
            conv9_2 = L.Convolution2D(128, 256, ksize = 3, stride = 1, pad = 0),

            #bn9_2 = L.BatchNormalization(256),

            conv9_2_mbox_loc = L.Convolution2D(256, common_params.num_of_offset_dims * common_params.num_boxes[5], ksize = 3, stride = 1, pad = 1),
            conv9_2_mbox_cls = L.Convolution2D(256, common_params.num_of_classes * common_params.num_boxes[5], ksize = 3, stride = 1, pad = 1),
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
        #k1 = F.relu(self.bn4_3(self.conv4_3(k1), test = not self.train))

        Loc1 = self.conv4_3_mbox_loc(self.bn4_3(k1, test = not self.train)) # Box_Estimator_1
        Cls1 = self.conv4_3_mbox_cls(self.bn4_3(k1, test = not self.train)) # Class_Classifier_1


        #L1 = self.conv4_3_norm_mbox_loc(F.local_response_normalization(k1))  # Box_Estimator_1
        #C1 = self.conv4_3_norm_mbox_cls(F.local_response_normalization(k1)) # Class_Classifier_1

        k2 = F.max_pooling_2d(k1, ksize = 2, stride = 2, pad = 0) # pooling_4

        k2 = F.relu(self.conv5_1(k2))
        k2 = F.relu(self.conv5_2(k2))
        k2 = F.relu(self.conv5_3(k2))

        k2 = F.max_pooling_2d(k2, ksize = 3, stride = 1, pad = 1) # pooling_5

        k2 = F.relu(self.fc6(k2))
        k2 = F.relu(self.fc7(k2))
        #k2 = F.relu(self.bn7(self.fc7(k2), test = not self.train))

        Loc2 = self.fc7_mbox_loc(k2) # Box_Estimator_2
        Cls2 = self.fc7_mbox_cls(k2) # Class_Classifier_2

        k3 = F.relu(self.conv6_1(k2))
        k3 = F.relu(self.conv6_2(k3))
        #k3 = F.relu(self.bn6_2(self.conv6_2(k3), test = not self.train))

        Loc3 = self.conv6_2_mbox_loc(k3) # Box_Estimator_3
        Cls3 = self.conv6_2_mbox_cls(k3) # Class_Classifier_3

        k4 = F.relu(self.conv7_1(k3))
        k4 = F.relu(self.conv7_2(k4))
        #k4 = F.relu(self.bn7_2(self.conv7_2(k4), test = not self.train))

        Loc4 = self.conv7_2_mbox_loc(k4) # Box_Estimator_4
        Cls4 = self.conv7_2_mbox_cls(k4) # Class_Classifier_4

        k5 = F.relu(self.conv8_1(k4))
        k5 = F.relu(self.conv8_2(k5))
        #k5 = F.relu(self.bn8_2(self.conv8_2(k5), test = not self.train))

        Loc5 = self.conv8_2_mbox_loc(k5) # Box_Estimator_5
        Cls5 = self.conv8_2_mbox_cls(k5) # Class_Classifier_5

        k6 = F.relu(self.conv9_1(k5))
        k6 = F.relu(self.conv9_2(k6))
        #k6 = F.relu(self.bn9_2(self.conv9_2(k6), test = not self.train))

        Loc6 = self.conv9_2_mbox_loc(k6) # Box_Estimator_6
        Cls6 = self.conv9_2_mbox_cls(k6) # Class_Classifier_6

        if self.train:
            Loc1 = F.transpose(Loc1, [0, 2, 3, 1])
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

            return (Loc1, Cls1, Loc2, Cls2, Loc3, Cls3, Loc4, Cls4, Loc5, Cls5, Loc6, Cls6)

        else:
            return (Loc1, Cls1, Loc2, Cls2, Loc3, Cls3, Loc4, Cls4, Loc5, Cls5, Loc6, Cls6)
