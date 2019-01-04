import matplotlib
import sys
import math
from os import path

# Ignore first 1000 itr
IGNORE_FIRST = True

# interval
INTERVAL = 500

def main():
    if len(sys.argv) != 3:
        print("Usage: python {} loss_dir_path save_flag(1/0)".format(sys.argv[0]))
        exit(1)

    #input_path = '/home/ryorsk/SSDsegmentation/models/2018-06-30@21-07-06/MTDSSD_loss/'
    input_path = sys.argv[1]
    save_flag = int(sys.argv[2])

    if save_flag == 1:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cnt_chk_cls = 0
    cnt_chk_loc = 0
    cnt_chk_seg = 0

    cls_loss, loc_loss, seg_loss, cls_loss_r, loc_loss_r, seg_loss_r = [], [], [], [], [], []
    for l in open(path.join(input_path, 'loss_cls.txt')).readlines():
        data = l[:-1].split(' ')
        cls_loss += [float(data[0])]
    for i in range(len(cls_loss)):
        if i % INTERVAL == 0:
            if i == 0 and IGNORE_FIRST: continue
            cnt_chk_cls += 1
            cls_loss_r.append(cls_loss[i])


    for l in open(path.join(input_path, 'loss_loc.txt')).readlines():
        data = l[:-1].split(' ')
        loc_loss += [float(data[0])]
    for i in range(len(loc_loss)):
        if i % INTERVAL == 0:
            if i == 0 and IGNORE_FIRST: continue
            cnt_chk_loc += 1
            loc_loss_r.append(loc_loss[i])

    for l in open(path.join(input_path, 'loss_seg.txt')).readlines():
        data = l[:-1].split(' ')
        seg_loss += [float(data[0])]
    for i in range(len(seg_loss)):
        if i % INTERVAL == 0:
            if i == 0 and IGNORE_FIRST: continue
            cnt_chk_seg += 1
            seg_loss_r.append(seg_loss[i])

    if path.exists(path.join(input_path, 'loss_seg_aux.txt')):
        cnt_chk_aux = 0
        aux_loss = []
        aux_loss_r = []
        for l in open(path.join(input_path, 'loss_seg_aux.txt')).readlines():
            data = l[:-1].split(' ')
            aux_loss += [float(data[0])]
        for i in range(len(aux_loss)):
            if i % INTERVAL == 0:
                if i == 0 and IGNORE_FIRST: continue
                cnt_chk_aux += 1
                aux_loss_r.append(aux_loss[i])


    if not(cnt_chk_cls == cnt_chk_loc == cnt_chk_seg):
        print("Error: Number of lines is mismatched. cls:{} loc:{} seg:{}".format(cnt_chk_cls, cnt_chk_loc, cnt_chk_seg))

    elif path.exists(path.join(input_path, 'loss_seg_aux.txt')) and not(cnt_chk_cls == cnt_chk_loc == cnt_chk_seg == cnt_chk_aux):
        print("Error: Number of lines is mismatched. cls:{} loc:{} seg:{} aux:{}".format(cnt_chk_cls, cnt_chk_loc, cnt_chk_seg. cnt_chk_aux))

    x = []
    for l in range(1, cnt_chk_cls+1):
        if l == 0 and IGNORE_FIRST: continue
        x.append(l * INTERVAL)

    # plot

    lines = plt.plot(x, cls_loss_r, 'k')
    plt.setp(lines, color='b', linewidth=1.0)
    plt.ylabel('Class Loss')
    plt.xlabel('Iteration')
    if save_flag == 1:
        plt.savefig(path.join(input_path, "loss_cls.pdf"))
    else:
        plt.show()
    plt.clf()

    lines = plt.plot(x, loc_loss_r, 'g')
    plt.setp(lines, color='b', linewidth=1.0)
    plt.ylabel('Localization Loss')
    plt.xlabel('Iteration')
    if save_flag == 1:
        plt.savefig(path.join(input_path, "loss_loc.pdf"))
    else:
        plt.show()
    plt.clf()

    lines = plt.plot(x, seg_loss_r, 'b')
    plt.setp(lines, color='b', linewidth=1.0)
    plt.ylabel('Segmentation Loss')
    plt.xlabel('Iteration')
    if save_flag == 1:
        plt.savefig(path.join(input_path, "loss_seg.pdf"))
    else:
        plt.show()
    plt.clf()


    if path.exists(path.join(input_path, 'loss_seg_aux.txt')):
        lines = plt.plot(x, aux_loss_r, 'b')
        plt.setp(lines, color='b', linewidth=1.0)
        plt.ylabel('Auxiliary Loss')
        plt.xlabel('Iteration')
        if save_flag == 1:
            plt.savefig(path.join(input_path, "loss_seg_aux.pdf"))
        else:
            plt.show()
        plt.clf()

    plt.close()

if __name__ == '__main__':
    main()
