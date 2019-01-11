# -*- coding: utf-8 -*-
# python consistency_evaluation.py --image /Users/ryorsk/Documents/MIRU2018/experiments/results_ARCdataset_epc150/segimage_large/  --teach /Volumes/External/arcdataset/public/ARCdataset/test_known/boundingbox/ --result /Users/ryorsk/Documents/MIRU2018/experiments/results_ARCdataset_epc150/score/ --segresult /Users/ryorsk/Documents/MIRU2018/experiments/results_ARCdataset_epc150/segresult_gray/
import os
from os import path
import numpy
import cv2
import glob
import matplotlib.pyplot as plt
from pylab import *
import argparse

import myfunc

parser = argparse.ArgumentParser()
parser.add_argument('--image', '-i', type = str, default = './image/', help = 'Path of test images dir')
parser.add_argument('--teach', '-t', type = str, default = './teach/', help = 'Path of ground truth b-boxes dir')
parser.add_argument('--result', '-r', type = str, default = './results/', help = 'Path of detection results dir')
parser.add_argument('--segresult', '-s', type = str, default = './results/', help = 'Path of segmentation(grayscale) results dir')
args = parser.parse_args()

image_path = args.image
teach_path = args.teach
result_path = args.result
segresult_path = args.segresult

##### PLEASE CHANGE AS NECESSARY #################

# evaluation output(consistency_result.txt, consistency_confmat) path
eval_result_path = result_path + "/eval/"

# image extension(png, jpg, bmp)
IMAGE_EXT = "png"

# segmentation result image extension(png, jpg, bmp)
SEGRESULT_EXT = "png"

# wait time of cv2.waitKey (if 0 then no wait)
WAITTIME = 0

# if your detection results have a classlabel(e.g.: DVD, avery_binder...), set 1
LABEL_FLAG = 1

# if your detection results are normalized, set 1
NORMALIZED = 0

#IOU Threshold
IOU_THRESH = 0.55

# if teach labels have category and color classification results, set 1
# normally need not change
CAT_PASS_FLAG = 0

# normally need not change
THRESH = 0.35
NCLASS = 40

##################################################

COLOR_TABLE = myfunc.COLOR_TABLE

itemIDList = myfunc.itemIDList

def checkSegmentClass(box, det):
    # box: cropped segmentation results (grayscale, pixel=classID)
    # det: detection results(class, x1, y1, x2, y2)
    x1 = det[1]
    y1 = det[2]
    x2 = det[3]
    y2 = det[4]
    box = box[y1:y2, x1:x2]
    max_moment = 0
    major_class = 0
    box_original = box.copy()
    for i in range(0, NCLASS+1):
        box = box_original.copy()
        box[box!=i] = 0
        moment = cv2.countNonZero(box)
        if moment > max_moment:
            max_moment = moment
            major_class = i

    if WAITTIME != 0:
        box_original[box_original==major_class] = 255
        cv2.imshow("major_class:" + str(major_class), box_original)
        cv2.waitKey(WAITTIME*5)
        cv2.destroyWindow("major_class:" + str(major_class))
    return major_class

def consist_matching(file_list):
    consist_true = 0
    consist_false = 0
    for file_path in file_list:
        #print(file_path)
        file_name, ext = path.splitext( path.basename(file_path) )
        file_name = file_name.replace("res", "")

        result_data = myfunc.readTxt(file_path, "result", label_flag=LABEL_FLAG)
        teach_data = myfunc.readTxt(teach_path + file_name + ".txt", "teach", cat_pass_flag=CAT_PASS_FLAG)

        img = cv2.imread(image_path + file_name + "." + IMAGE_EXT)
        segment_img = cv2.imread(segresult_path + file_name + "." + SEGRESULT_EXT, cv2.IMREAD_GRAYSCALE)
        height, width = img.shape[:2]
        img_backup = img.copy()

        if img.shape[:2] != segment_img.shape[:2]:
            segment_img = cv2.resize(segment_img, (width, height), interpolation=cv2.INTER_NEAREST)
        segment_img = segment_img.astype(np.uint8)

        # Convert to Full(teach) coordinates
        for i in range(0, len(teach_data)):
            teach_data[i] = myfunc.convNormalizedCord(teach_data[i], height, width)

        # Convert for result coordinates
        for i in range(0, len(result_data)):
            result_data[i] = myfunc.convResCord(result_data[i], height, width, NORMALIZED)

        # search box which have highest IoU value
        for j in range(0, len(result_data)):
            max_IoU = 0.0
            max_index = 0
            for i in range(0, len(teach_data)):
                iou = myfunc.getIOU(teach_data[i][1:], result_data[j][1:])
                if(max_IoU < iou) and (iou <= 1.0):
                    max_IoU = iou
                    max_index = i
                if WAITTIME != 0:
                    img = img_backup.copy()
                    img = myfunc.drawBB(img, teach_data[i])
                    img = myfunc.drawBB(img, result_data[j])
                    cv2.imshow("", img)
                    cv2.waitKey(WAITTIME)

            # found
            img = img_backup.copy()
            if(max_IoU > THRESH):
                success = (teach_data[max_index][0] == result_data[j][0])
                if WAITTIME != 0:
                    img = myfunc.drawBB(img, teach_data[max_index])
                    img = myfunc.drawBB(img, result_data[j])
                    cv2.putText(img, "Max IoU:" + str(max_IoU), (0, 25) , cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255), 2)
                    cv2.putText(img, "Class Match:" + str(success), (0, 50) , cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255), 2)
                major_class = checkSegmentClass(segment_img.copy(), result_data[j])
                if major_class == result_data[j][0]:
                    consist_true += 1
                else:
                    consist_false += 1

            else:
                if WAITTIME != 0:
                    img = myfunc.drawBB(img, result_data[j])
                    cv2.putText(img, "No Matching", (0, 25) , cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255), 2)
            if WAITTIME != 0:
                cv2.imshow("", img)
                cv2.waitKey(WAITTIME*5)

    print("Total consistency result")
    print("Det==Seg:" + str(consist_true))
    print("Det!=Seg:" + str(consist_false))
    print("Consistency rate:" + str(float(consist_true) / (float(consist_true) + float(consist_false))))

    if WAITTIME != 0:
        cv2.destroyAllWindows()

def main():
    # IoU matching
    file_list = glob.glob(result_path + "*.txt")
    if len(file_list) == 0:
        print("[Error] Detection results file list is empty. Check this dir:" + result_path)
        file_list.sort()
    else:
        print(result_path)
        consist_matching(file_list)

if __name__ == '__main__':
    main()
