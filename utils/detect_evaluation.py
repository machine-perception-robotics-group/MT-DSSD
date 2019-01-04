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
args = parser.parse_args()

image_path = args.image
teach_path = args.teach
result_path = args.result

##### PLEASE CHANGE AS NECESSARY #################

# IoU matching results path
match_result_path = result_path + "/matchingResults/"

# evaluation output(totalresult, confusion_matrix) path
eval_result_path = result_path + "/eval/"

# image extension(png, jpg, bmp)
IMAGE_EXT = "png"

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

def matching(file_list):
    for file_path in file_list:
        print(file_path)
        file_name, ext = path.splitext( path.basename(file_path) )
        file_name = file_name.replace("res", "")

        result_data = myfunc.readTxt(file_path, "result", label_flag=LABEL_FLAG)
        teach_data = myfunc.readTxt(teach_path + file_name + ".txt", "teach", cat_pass_flag=CAT_PASS_FLAG)
        img = cv2.imread(image_path + file_name + "." + IMAGE_EXT)
        height, width = img.shape[:2]
        img_backup = img.copy()

        # Convert to Full(teach) coordinates
        for i in range(0, len(teach_data)):
            teach_data[i] = myfunc.convNormalizedCord(teach_data[i], height, width)

        # Convert for result coordinates
        for i in range(0, len(result_data)):
            result_data[i] = myfunc.convResCord(result_data[i], height, width, NORMALIZED)

        # init result lists
        hit = [False] * len(teach_data)
        success = [False] * len(result_data)
        max_IoU_list = [0] * len(result_data)
        true_category = [0] * len(result_data)

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
            max_IoU_list[j] = max_IoU
            img = img_backup.copy()
            if(max_IoU > THRESH):
                hit[max_index] = True
                success[j] = (teach_data[max_index][0] == result_data[j][0])
                true_category[j] = teach_data[max_index][0]
                if WAITTIME != 0:
                    img = myfunc.drawBB(img, teach_data[max_index])
                    img = myfunc.drawBB(img, result_data[j])
                    cv2.putText(img, "Max IoU:" + str(max_IoU), (0, 25) , cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255), 2)
                    cv2.putText(img, "Class Match:" + str(success[j]), (0, 50) , cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255), 2)
            else:
                if WAITTIME != 0:
                    img = myfunc.drawBB(img, result_data[j])
                    cv2.putText(img, "No Matching", (0, 25) , cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255), 2)
            if WAITTIME != 0:
                cv2.imshow("", img)
                cv2.waitKey(WAITTIME*5)

        print("Result")
        print("Matching Success")
        for i in range(0, len(result_data)):
            if(success[i] == True):
                print(str(result_data[i]))

        print("Matching Failed")
        for i in range(0, len(result_data)):
            if(success[i] == False):
                print(str(result_data[i]) + str(max_IoU_list[i]))

        print("No Match")
        for i in range(0, len(teach_data)):
            if(hit[i] == False):
                print(str(teach_data[i]))

        print("File output")
        f = open(result_path + "matchingResults/" + file_name + ".txt", 'w')
        for i in range(0, len(result_data)):
            write_data = str(result_data[i][0]) + " " + str(true_category[i]) + " " + str(max_IoU_list[i]) + '\n'
            f.writelines(write_data)
        # miss boxes
        for i in range(0, len(teach_data)):
            if(hit[i] == False):
                write_data = "0" + " " + str(teach_data[i][0]) + " 0.0" + '\n'
                f.writelines(write_data)
        f.close()

    if WAITTIME != 0:
        cv2.destroyAllWindows()

def evaluate(file_list):
    conv_ID_table = [i for i in range(0, NCLASS+1)]

    #if active this list, ID order change to ARC Official List order.
    """
    conv_ID_table = [
    0, 5, 2, 7, 39, 8, 9, 14, 13, 19, 15,
    32, 27, 34, 33, 29, 24, 18, 23, 11, 4,
    20, 38, 28, 31, 1, 6, 12, 35, 10, 22,
    21, 36, 25, 3, 30, 26, 40, 37, 16, 17, 41]
    """

    total_boxes = 0             # number of all(detected + not detected) boxes
    total_true_boxes = 0        # number of true class boxes
    total_detected_boxes = 0    # number of detected boxes
    total_false_boxes = 0       # number of false class boxes
    total_undetected_boxes = 0  # number of not detected boxes
    total_IoU = 0.0

    confusion_mat = [[0 for i in range(NCLASS)] for j in range(NCLASS)]

    for filePath in file_list:
        boxes = 0
        true_boxes = 0
        detected_boxes = 0
        false_boxes = 0
        undetected_boxes = 0

        result_data = myfunc.readTxt(filePath, "evaluate")

        for i in range(0, len(result_data)):
            boxes += 1
            if(result_data[i][0] != '0' and result_data[i][1] != '0'):
                #detected
                detected_boxes += 1
                total_IoU += float(result_data[i][2])
                confusion_mat[conv_ID_table[int(result_data[i][1])]-1][conv_ID_table[int(result_data[i][0])]-1] += 1
                if(float(result_data[i][2]) >= IOU_THRESH):
                    #IOU >= Threshold
                    if (result_data[i][0] == result_data[i][1]):
                        #true class
                        true_boxes += 1
                    else:
                        #false class
                        false_boxes += 1
                else:
                    #low IOU
                    false_boxes += 1
            else:
                #not detected
                undetected_boxes += 1

        if boxes != (detected_boxes + undetected_boxes):
            print("[Error] missmatch: boxes != detected_boxes + undetected_boxes")
            print(boxes, detected_boxes, undetected_boxes)
            sys.exit()
        if detected_boxes != (true_boxes + false_boxes):
            print("[Error] missmatch: detected_boxes != true_boxes + false_boxes")
            print(detected_boxes, true_boxes, false_boxes)
            sys.exit()

        total_boxes += boxes
        total_true_boxes += true_boxes
        total_detected_boxes += detected_boxes
        total_false_boxes += false_boxes
        total_undetected_boxes += undetected_boxes

    print("Total Result")
    print("Matching Rate: "+str(float(total_true_boxes) / float(total_detected_boxes)))
    print("Miss(No detection box) Rate: "+str(float(total_undetected_boxes) / float(total_boxes)))
    print("Mean IoU:"+ str(total_IoU / total_detected_boxes))

    #confusion_matrix normalization
    normalized_matrix = []
    for i in confusion_mat:
        a = 0
        temp_matrix = []
        a = sum(i,0)
        for j in i:
            if a == 0:
                temp_matrix.append(0.0)
            else:
                temp_matrix.append(float(j) / float(a))
        normalized_matrix.append(temp_matrix)

    #draw confusion_matrix
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    res = ax.imshow(array(normalized_matrix), cmap=cm.jet, interpolation='nearest')
    cb = fig.colorbar(res)
    cb.ax.set_yticklabels([str(i)+'%' for i in range(0, 101, 10)])
    confusion_mat = numpy.array(confusion_mat)
    width, height = confusion_mat.shape
    item_list = [str(i) for i in range(1, NCLASS+1)]
    plt.xticks(range(width), item_list[:width],rotation=90)
    plt.yticks(range(height), item_list[:height])
    plt.tick_params(labelsize=7)
    plt.subplots_adjust(left=0.05, bottom=0.10, right=0.95, top=0.95)
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")

    for i in range(0, NCLASS):
        print(str(i+1) + ": " + str(normalized_matrix[i][i]))

    #file output
    f = open(eval_result_path + "/totalresult.txt", 'w')
    f.writelines("Total Result" + '\n')
    f.writelines("Matching Rate: "+str(float(total_true_boxes) / float(total_detected_boxes)) + '\n')
    f.writelines("Miss(No detection box) Rate: "+str(float(total_undetected_boxes) / float(total_boxes)) + '\n')
    f.writelines("Mean IoU:"+ str(total_IoU / total_detected_boxes) + '\n')
    for i in range(0, NCLASS):
        f.writelines(str(i+1) + ": " + str(normalized_matrix[i][i]) + '\n')
    f.close()

    savefig(eval_result_path + "/confusion_matrix.pdf", format="pdf")
    savefig(eval_result_path + "/confusion_matrix.png", format="png")


def main():
    # mkdir for Matching results
    if not os.path.exists(result_path + "/matchingResults"): os.mkdir(result_path + "/matchingResults")
    if not os.path.exists(result_path + "/eval"): os.mkdir(result_path + "/eval")

    # IoU matching
    file_list = glob.glob(result_path + "*.txt")
    if len(file_list) == 0:
        print("[Error] Detection results file list is empty. Check this dir:" + result_path)
        file_list.sort()
    else:
        matching(file_list)

    # evaluate and output
    file_list = glob.glob(match_result_path + "/*.txt")
    if len(file_list) == 0:
        print("[Error] Evaluation file list is empty. Check this dir:" + match_result_path)
    else:
        file_list.sort()
        evaluate(file_list)

if __name__ == '__main__':
    main()
