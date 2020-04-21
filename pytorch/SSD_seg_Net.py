#! /usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import common_params


class SSDNet(nn.Module):

    def __init__(self):
        super(SSDNet, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1)

        self.maxpool1 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)

        self.maxpool2 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1)

        self.maxpool3 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, ceil_mode = True)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1)

        self.maxpool4 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1)

        self.maxpool5 = nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1)

        self.fc6 = nn.Conv2d(512, 1024, kernel_size = 3, stride = 1, padding = 1)
        self.fc7 = nn.Conv2d(1024, 1024, kernel_size = 1, stride = 1, padding = 0)

        self.conv6_1 = nn.Conv2d(1024, 256, kernel_size = 1, stride = 1, padding = 0)
        self.conv6_2 = nn.Conv2d(256, 512, kernel_size = 3, stride = 2, padding = 1)
        #self.bn_conv6 = nn.BatchNorm2d(512)

        self.conv7_1 = nn.Conv2d(512, 128, kernel_size = 1, stride = 1, padding = 0)
        self.conv7_2 = nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding = 1)

        self.conv8_1 = nn.Conv2d(256, 128, kernel_size = 1, stride = 1, padding = 0)
        self.conv8_2 = nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 0)

        self.conv9_1 = nn.Conv2d(256, 128, kernel_size = 1, stride = 1, padding = 0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 0)

        self.conv9_2_mbox_loc = nn.Conv2d(256, common_params.num_of_offset_dims * common_params.num_boxes[5], kernel_size = 3, stride = 1, padding = 1)
        self.conv9_2_mbox_cls = nn.Conv2d(256, common_params.num_of_classes * common_params.num_boxes[5], kernel_size = 3, stride = 1, padding = 1)

        self.deconv9_2_1 = nn.ConvTranspose2d(256, common_params.num_of_classes, kernel_size = 10, stride = 1, padding = 0) #10×10
        self.deconv9_2_2 = nn.ConvTranspose2d(common_params.num_of_classes, common_params.num_of_classes, kernel_size = 11, stride = 8, padding = 4) #75×75
        self.deconv9_2_3 = nn.ConvTranspose2d(common_params.num_of_classes, common_params.num_of_classes, kernel_size = 6, stride = 4, padding = 1) #300×300

        self.deconv1 = nn.ConvTranspose2d(256, 256, kernel_size = 3, stride = 1, padding = 0) #3×3

        self.conv8_2_mbox_loc = nn.Conv2d(256, common_params.num_of_offset_dims * common_params.num_boxes[4], kernel_size = 3, stride = 1, padding = 1)
        self.conv8_2_mbox_cls = nn.Conv2d(256, common_params.num_of_classes * common_params.num_boxes[4], kernel_size = 3, stride = 1, padding = 1)

        self.deconv8_2_1 = nn.ConvTranspose2d(256, common_params.num_of_classes, kernel_size = 9, stride = 6, padding = 1) #19×19
        self.deconv8_2_2 = nn.ConvTranspose2d(common_params.num_of_classes, common_params.num_of_classes, kernel_size = 7, stride = 4, padding = 2) #75×75
        self.deconv8_2_3 = nn.ConvTranspose2d(common_params.num_of_classes, common_params.num_of_classes, kernel_size = 6, stride = 4, padding = 1) #300×300

        self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size = 3, stride = 2, padding = 1) #5×5

        self.conv7_2_mbox_loc = nn.Conv2d(256, common_params.num_of_offset_dims * common_params.num_boxes[3], kernel_size = 3, stride = 1, padding = 1)
        self.conv7_2_mbox_cls = nn.Conv2d(256, common_params.num_of_classes * common_params.num_boxes[3], kernel_size = 3, stride = 1, padding = 1)

        self.deconv7_2_1 = nn.ConvTranspose2d(256, common_params.num_of_classes, kernel_size = 7, stride = 4, padding = 2) #19×19
        self.deconv7_2_2 = nn.ConvTranspose2d(common_params.num_of_classes, common_params.num_of_classes, kernel_size = 7, stride = 4, padding = 2) #75×75
        self.deconv7_2_3 = nn.ConvTranspose2d(common_params.num_of_classes, common_params.num_of_classes, kernel_size = 6, stride = 4, padding = 1) #300×300

        self.deconv3 = nn.ConvTranspose2d(256, 512, kernel_size = 4, stride = 2, padding = 1) #10×10

        self.conv6_2_mbox_loc = nn.Conv2d(512, common_params.num_of_offset_dims * common_params.num_boxes[2], kernel_size = 3, stride = 1, padding = 1)
        self.conv6_2_mbox_cls = nn.Conv2d(512, common_params.num_of_classes * common_params.num_boxes[2], kernel_size = 3, stride = 1, padding = 1)

        self.deconv6_2_1 = nn.ConvTranspose2d(512, common_params.num_of_classes, kernel_size = 6, stride = 4, padding = 2) #38×38
        self.deconv6_2_2 = nn.ConvTranspose2d(common_params.num_of_classes, common_params.num_of_classes, kernel_size = 3, stride = 2, padding = 1) #75×75
        self.deconv6_2_3 = nn.ConvTranspose2d(common_params.num_of_classes, common_params.num_of_classes, kernel_size = 6, stride = 4, padding = 1) #300×300

        self.deconv4 = nn.ConvTranspose2d(512, 1024, kernel_size = 3, stride = 2, padding = 1) #19×19

        self.fc7_mbox_loc = nn.Conv2d(1024, common_params.num_of_offset_dims * common_params.num_boxes[1], kernel_size = 3, stride = 1, padding = 1)
        self.fc7_mbox_cls = nn.Conv2d(1024, common_params.num_of_classes * common_params.num_boxes[1], kernel_size = 3, stride = 1, padding = 1)

        self.deconvfc7_1 = nn.ConvTranspose2d(1024, common_params.num_of_classes, kernel_size = 4, stride = 2, padding = 1) #38×38
        self.deconvfc7_2 = nn.ConvTranspose2d(common_params.num_of_classes, common_params.num_of_classes, kernel_size = 3, stride = 2, padding = 1) #75×75
        self.deconvfc7_3 = nn.ConvTranspose2d(common_params.num_of_classes, common_params.num_of_classes, kernel_size = 6, stride = 4, padding = 1) #300×300

        self.deconv5 = nn.ConvTranspose2d(1024, 512, kernel_size = 4, stride = 2, padding = 1) #38×38

        #self.bn4_3 = nn.BatchNorm2d(512)

        self.conv4_3_norm_mbox_loc = nn.Conv2d(512, common_params.num_of_offset_dims * common_params.num_boxes[0], kernel_size = 3, stride = 1, padding = 1)
        self.conv4_3_norm_mbox_cls = nn.Conv2d(512, common_params.num_of_classes * common_params.num_boxes[0], kernel_size = 3, stride = 1, padding = 1)

        self.deconv4_3_1 = nn.ConvTranspose2d(512, common_params.num_of_classes, kernel_size = 3, stride = 2, padding = 1) #75×75
        self.deconv4_3_2 = nn.ConvTranspose2d(common_params.num_of_classes, common_params.num_of_classes, kernel_size = 4, stride = 2, padding = 1) #150×150
        self.deconv4_3_3 = nn.ConvTranspose2d(common_params.num_of_classes, common_params.num_of_classes, kernel_size = 4, stride = 2, padding = 1)  #300×300

        self.segconv = nn.Conv2d(common_params.num_of_classes*6, common_params.num_of_classes, kernel_size = 1, stride = 1, padding = 0)


    def forward(self, x, train=False):

        k1 = F.relu(self.conv1_1(x))
        k1 = F.relu(self.conv1_2(k1))

        k1 = self.maxpool1(k1)

        k1 = F.relu(self.conv2_1(k1))
        k1 = F.relu(self.conv2_2(k1))

        k1 = self.maxpool2(k1)

        k1 = F.relu(self.conv3_1(k1))
        k1 = F.relu(self.conv3_2(k1))
        k1 = F.relu(self.conv3_3(k1))

        k1 = self.maxpool3(k1)

        k1 = F.relu(self.conv4_1(k1))
        k1 = F.relu(self.conv4_2(k1))
        k1 = F.relu(self.conv4_3(k1))

        k2 = self.maxpool4(k1)

        k2 = F.relu(self.conv5_1(k2))
        k2 = F.relu(self.conv5_2(k2))
        k2 = F.relu(self.conv5_3(k2))

        k2 = self.maxpool5(k2)

        k2 = F.relu(self.fc6(k2))
        k2 = F.relu(self.fc7(k2))

        k3 = F.relu(self.conv6_1(k2))
        k3 = F.relu(self.conv6_2(k3))

        k4 = F.relu(self.conv7_1(k3))
        k4 = F.relu(self.conv7_2(k4))

        k5 = F.relu(self.conv8_1(k4))
        k5 = F.relu(self.conv8_2(k5))

        k6 = F.relu(self.conv9_1(k5))
        k6 = F.relu(self.conv9_2(k6))

        Loc6 = self.conv9_2_mbox_loc(k6)  # Box_Estimator_6
        Cls6 = self.conv9_2_mbox_cls(k6) # Class_Classifier_6

        Seg1 = self.deconv9_2_1(k6)
        Seg1 = self.deconv9_2_2(Seg1)
        Seg1 = self.deconv9_2_3(Seg1)

        y5 = self.deconv1(k6) #3×3

        yy5 = F.relu(y5 + k5)

        Loc5 = self.conv8_2_mbox_loc(yy5)  # Box_Estimator_5
        Cls5 = self.conv8_2_mbox_cls(yy5) # Class_Classifier_5

        Seg2 = self.deconv8_2_1(yy5)
        Seg2 = self.deconv8_2_2(Seg2)
        Seg2 = self.deconv8_2_3(Seg2)

        y4 = self.deconv2(yy5)

        yy4 = F.relu(y4 + k4)

        Loc4 = self.conv7_2_mbox_loc(yy4)  # Box_Estimator_4
        Cls4 = self.conv7_2_mbox_cls(yy4) # Class_Classifier_4

        Seg3 = self.deconv7_2_1(yy4)
        Seg3 = self.deconv7_2_2(Seg3)
        Seg3 = self.deconv7_2_3(Seg3)

        y3 = self.deconv3(yy4)

        yy3 = F.relu(y3 + k3)

        Loc3 = self.conv6_2_mbox_loc(yy3)  # Box_Estimator_3
        Cls3 = self.conv6_2_mbox_cls(yy3) # Class_Classifier_3

        Seg4 = self.deconv6_2_1(yy3)
        Seg4 = self.deconv6_2_2(Seg4)
        Seg4 = self.deconv6_2_3(Seg4)

        y2 = self.deconv4(yy3)

        yy2 = F.relu(y2 + k2)

        Loc2 = self.fc7_mbox_loc(yy2)  # Box_Estimator_2
        Cls2 = self.fc7_mbox_cls(yy2) # Class_Classifier_2

        Seg5 = self.deconvfc7_1(yy2)
        Seg5 = self.deconvfc7_2(Seg5)
        Seg5 = self.deconvfc7_3(Seg5)

        y1 = self.deconv5(yy2)

        yy1 = F.relu(y1 + k1)

        #Loc1 = self.conv4_3_norm_mbox_loc(self.bn4_3(yy1))  # Box_Estimator_1
        #Cls1 = self.conv4_3_norm_mbox_cls(self.bn4_3(yy1)) # Class_Classifier_1
        Loc1 = self.conv4_3_norm_mbox_loc(yy1)  # Box_Estimator_1
        Cls1 = self.conv4_3_norm_mbox_cls(yy1) # Class_Classifier_1

        Seg6 = self.deconv4_3_1(yy1) #75×75
        Seg6 = self.deconv4_3_2(Seg6) #150×150
        Seg6 = self.deconv4_3_3(Seg6) #300×300

        Seg = torch.cat([Seg1, Seg2, Seg3, Seg4, Seg5, Seg6], dim = 1)
        #Seg = Seg1 + Seg2 + Seg3 + Seg4 + Seg5 + Seg6

        Seg = self.segconv(Seg)

        if train:
            Loc1 = Loc1.permute(0, 2, 3, 1) #(バッチ数,高さ,幅,チャンネル数)
            Cls1 = Cls1.permute(0, 2, 3, 1)

            Loc2 = Loc2.permute(0, 2, 3, 1)
            Cls2 = Cls2.permute(0, 2, 3, 1)

            Loc3 = Loc3.permute(0, 2, 3, 1)
            Cls3 = Cls3.permute(0, 2, 3, 1)

            Loc4 = Loc4.permute(0, 2, 3, 1)
            Cls4 = Cls4.permute(0, 2, 3, 1)

            Loc5 = Loc5.permute(0, 2, 3, 1)
            Cls5 = Cls5.permute(0, 2, 3, 1)

            Loc6 = Loc6.permute(0, 2, 3, 1)
            Cls6 = Cls6.permute(0, 2, 3, 1)

            Loc1 = Loc1.contiguous()
            Loc2 = Loc2.contiguous()
            Loc3 = Loc3.contiguous()
            Loc4 = Loc4.contiguous()
            Loc5 = Loc5.contiguous()
            Loc6 = Loc6.contiguous()

            Cls1 = Cls1.contiguous()
            Cls2 = Cls2.contiguous()
            Cls3 = Cls3.contiguous()
            Cls4 = Cls4.contiguous()
            Cls5 = Cls5.contiguous()
            Cls6 = Cls6.contiguous()

            Loc1 = Loc1.view(Loc1.data.shape[0] * Loc1.data.shape[1] * Loc1.data.shape[2] * common_params.num_boxes[0], int(Loc1.data.shape[3] / common_params.num_boxes[0]))
            Cls1 = Cls1.view(Cls1.data.shape[0] * Cls1.data.shape[1] * Cls1.data.shape[2] * common_params.num_boxes[0], int(Cls1.data.shape[3] / common_params.num_boxes[0]))

            Loc2 = Loc2.view(Loc2.data.shape[0] * Loc2.data.shape[1] * Loc2.data.shape[2] * common_params.num_boxes[1], int(Loc2.data.shape[3] / common_params.num_boxes[1]))
            Cls2 = Cls2.view(Cls2.data.shape[0] * Cls2.data.shape[1] * Cls2.data.shape[2] * common_params.num_boxes[1], int(Cls2.data.shape[3] / common_params.num_boxes[1]))

            Loc3 = Loc3.view(Loc3.data.shape[0] * Loc3.data.shape[1] * Loc3.data.shape[2] * common_params.num_boxes[2], int(Loc3.data.shape[3] / common_params.num_boxes[2]))
            Cls3 = Cls3.view(Cls3.data.shape[0] * Cls3.data.shape[1] * Cls3.data.shape[2] * common_params.num_boxes[2], int(Cls3.data.shape[3] / common_params.num_boxes[2]))

            Loc4 = Loc4.view(Loc4.data.shape[0] * Loc4.data.shape[1] * Loc4.data.shape[2] * common_params.num_boxes[3], int(Loc4.data.shape[3] / common_params.num_boxes[3]))
            Cls4 = Cls4.view(Cls4.data.shape[0] * Cls4.data.shape[1] * Cls4.data.shape[2] * common_params.num_boxes[3], int(Cls4.data.shape[3] / common_params.num_boxes[3]))

            Loc5 = Loc5.view(Loc5.data.shape[0] * Loc5.data.shape[1] * Loc5.data.shape[2] * common_params.num_boxes[4], int(Loc5.data.shape[3] / common_params.num_boxes[4]))
            Cls5 = Cls5.view(Cls5.data.shape[0] * Cls5.data.shape[1] * Cls5.data.shape[2] * common_params.num_boxes[4], int(Cls5.data.shape[3] / common_params.num_boxes[4]))

            Loc6 = Loc6.view(Loc6.data.shape[0] * Loc6.data.shape[1] * Loc6.data.shape[2] * common_params.num_boxes[5], int(Loc6.data.shape[3] / common_params.num_boxes[5]))
            Cls6 = Cls6.view(Cls6.data.shape[0] * Cls6.data.shape[1] * Cls6.data.shape[2] * common_params.num_boxes[5], int(Cls6.data.shape[3] / common_params.num_boxes[5]))

            return (Loc1, Cls1, Loc2, Cls2, Loc3, Cls3, Loc4, Cls4, Loc5, Cls5, Loc6, Cls6, Seg)

        else:
            return (Loc1, Cls1, Loc2, Cls2, Loc3, Cls3, Loc4, Cls4, Loc5, Cls5, Loc6, Cls6, Seg)
