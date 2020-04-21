#! /usr/bin/env python
# -*- coding: utf-8 -*-

from glob import glob
from os import path
import os
import sys
import argparse
import common_params


def normalImageListGeneration():

    fout = open('./img_name_list.txt', 'w')

    lists = glob(path.join(common_params.images_dir + '/train/boundingbox/' , '*'))
    print len(lists)

    for fl in lists:
        print fl
        items = fl.split('/')
        file_name, ext = os.path.splitext(items[-1])
        out_line = '{}\n'.format(file_name)
        fout.write(out_line)


def augmentedImageListGeneration():

    fout = open('./augimg_name_list.txt', 'w')

    lists = glob(path.join(common_params.images_dir + '/train/img_aug_param/', '*'))
    print len(lists)

    for fl in lists:
        print fl
        items = fl.split('/')
        file_name, ext = os.path.splitext(items[-1])
        out_line = '{}\n'.format(file_name)
        fout.write(out_line)


def segmentationImageListGeneration():

    fout = open('./segimg_name_list.txt', 'w')

    lists = glob(path.join(common_params.images_dir + '/train/segimg_aug_param/', '*'))
    print len(lists)

    for fl in lists:
        print fl
        items = fl.split('/')
        file_name, ext = os.path.splitext(items[-1])
        out_line = '{}\n'.format(file_name)
        fout.write(out_line)


def segmentationLabelListGeneration():

    fout = open('./seglabel_name_list.txt', 'w')

    lists = glob(path.join(common_params.images_dir + '/train/seglabel_aug_param/', '*'))
    print len(lists)

    for fl in lists:
        print fl
        items = fl.split('/')
        file_name, ext = os.path.splitext(items[-1])
        out_line = '{}\n'.format(file_name)
        fout.write(out_line)

def depthImageListGeneration():

    fout = open('./depth_name_list.txt', 'w')

    lists = glob(path.join(common_params.images_dir + '/train/depth_aug_param/', '*'))
    print len(lists)

    for fl in lists:
        print fl
        items = fl.split('/')
        file_name, ext = os.path.splitext(items[-1])
        out_line = '{}\n'.format(file_name)
        fout.write(out_line)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', '-M', type = int, default = 0, help = 'List generation mode')
args = parser.parse_args()

if args.mode == 0:
    normalImageListGeneration()
elif args.mode == 1:
    augmentedImageListGeneration()
elif args.mode == 2:
    segmentationImageListGeneration()
elif args.mode == 3:
    segmentationLabelListGeneration()
elif args.mode == 4:
    depthImageListGeneration()
