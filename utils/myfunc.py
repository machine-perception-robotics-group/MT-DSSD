import numpy
import cv2

COLOR_TABLE = [
[   0,    0,    0],
[  85,    0,    0],
[ 170,    0,    0],
[ 255,    0,    0],
[   0,   85,    0],
[  85,   85,    0],
[ 170,   85,    0],
[ 255,   85,    0],
[   0,  170,    0],
[  85,  170,    0],
[ 170,  170,    0],
[ 255,  170,    0],
[   0,  255,    0],
[  85,  255,    0],
[ 170,  255,    0],
[ 255,  255,    0],
[   0,    0,   85],
[  85,    0,   85],
[ 170,    0,   85],
[ 255,    0,   85],
[   0,   85,   85],
[  85,   85,   85],
[ 170,   85,   85],
[ 255,   85,   85],
[   0,  170,   85],
[  85,  170,   85],
[ 170,  170,   85],
[ 255,  170,   85],
[   0,  255,   85],
[  85,  255,   85],
[ 170,  225,   85],
[ 255,  255,   85],
[   0,    0,  170],
[  85,    0,  170],
[ 170,    0,  170],
[ 255,    0,  170],
[   0,   85,  170],
[  85,   85,  170],
[ 170,   85,  170],
[ 255,   85,  170],
[   0,  170,  170],
[ 255,  255,  255]]

itemIDList = [
"0 BG",
"1",
"2",
"3",
"4",
"5",
"6",
"7",
"8",
"9",
"10",
"11",
"12",
"13",
"14",
"15",
"16",
"17",
"18",
"19",
"20",
"21",
"22",
"23",
"24",
"25",
"26",
"27",
"28",
"29",
"30",
"31",
"32",
"33",
"34",
"35",
"36",
"37",
"38",
"39",
"40",
"41"]

def convNormalizedCord(data, height, width):
    x = float(data[1]) * width
    y = float(data[2]) * height
    w = float(data[3]) * width
    h = float(data[4]) * height
    x1 = x - (w/2.)
    y1 = y - (h/2.)
    x2 = x + (w/2.)
    y2 = y + (h/2.)
    return(int(data[0]), int(x1), int(y1), int(x2), int(y2))

def convResCord(data, height, width, normalized):
    if normalized == 1:
        data = convNormalizedCord(data, height, width)
    if int(data[0]) == 0:
        classID = 41
    else:
        classID = int(data[0])
    return(classID, int(data[1]), int(data[2]), int(data[3]), int(data[4]) )

def getIOU(boxA, boxB):
    # if the length of between boxAcenter and boxBcenter is too far, return 0
    center_boxA = numpy.array([(boxA[0] + boxA[2]) / 2.0, (boxA[1] + boxA[3]) / 2.0])
    center_boxB = numpy.array([(boxB[0] + boxB[2]) / 2.0, (boxB[1] + boxB[3]) / 2.0])
    if numpy.linalg.norm(center_boxA - center_boxB) >= 500:
        return 0

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = (xB - xA + 1) * (yB - yA + 1)
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = inter_area / float(boxA_area + boxB_area - inter_area)

    return iou

def readTxt(file_path, d_type, label_flag=0, cat_pass_flag=0):
    coordinate = []
    f = open(file_path, 'r')

    if f != None:
        for row in f:
            data = row.split()
            if(label_flag==1 and d_type == "result"):
                data = [data[0], data[2], data[3], data[4], data[5]]
            elif(d_type == "result"):
                data = [data[0], data[1], data[2], data[3], data[4]]
            elif(cat_pass_flag==1 and d_type == "teach"):
                data = [data[0], data[3], data[4], data[5], data[6]]
            elif(d_type == "teach" or d_type == "evaluate"):
                data = data
                # DO NOTHING
            else:
                print("[ERROR] Unexpected text data type:" + d_type)
                return 1
            coordinate.append(data)
        f.close()
        return coordinate
    else:
        print("[ERROR] Can't read:" + file_path)
        return 1

def drawBB(img, data):
    color = [ COLOR_TABLE[int(data[0])][2], COLOR_TABLE[int(data[0])][1], COLOR_TABLE[int(data[0])][0] ]
    height, width = img.shape[:2]
    x1 = data[1]
    y1 = data[2]
    x2 = data[3]
    y2 = data[4]
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    cv2.putText(img, itemIDList[int(data[0])], (x1, y1-2), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255), 2)
    cv2.putText(img, itemIDList[int(data[0])], (x1, y1), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,0), 1)
    return img
