#!/bin/sh
if [ "$1" = "" ]
then
    echo "Usage: sh mkdirs.sh <Your Dataset Path>"
else
    mkdir $1/train/positives
    mkdir $1/train/negatives
    mkdir $1/train/img_aug_param
    mkdir $1/train/seglabel_aug_param
fi
