import matplotlib
import sys
import math
from os import path
import numpy as np

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

    cls_loss = []
    if path.exists(path.join(input_path, 'loss_cls.txt')):
        for l in open(path.join(input_path, 'loss_cls.txt')).readlines():
            data = l[:-1].split(' ')
            value = float(data[0])
            if np.isnan(value) or np.isinf(value): value = 0.
            cls_loss.append(value)

    loc_loss = []
    if path.exists(path.join(input_path, 'loss_loc.txt')):
        for l in open(path.join(input_path, 'loss_loc.txt')).readlines():
            data = l[:-1].split(' ')
            value = float(data[0])
            if np.isnan(value) or np.isinf(value): value = 0.
            loc_loss.append(value)

    seg_loss = []
    if path.exists(path.join(input_path, 'loss_seg.txt')):
        for l in open(path.join(input_path, 'loss_seg.txt')).readlines():
            data = l[:-1].split(' ')
            value = float(data[0])
            if np.isnan(value) or np.isinf(value): value = 0.
            seg_loss.append(value)

    # convolve
    num_conv = 300
    b = np.ones(num_conv) / num_conv

    # plot
    if path.exists(path.join(input_path, 'loss_cls.txt')):
        x = range(0, len(cls_loss))
        lines = plt.plot(x, cls_loss)
        plt.setp(lines, color='c', linewidth=1.0, alpha=0.5)
        cls_loss_ave = np.convolve(cls_loss, b, mode='same')
        lines = plt.plot(x, cls_loss_ave)
        plt.setp(lines, color='b', linewidth=1.0)
        plt.ylim(-0.1, 1.5)
        plt.ylabel('Class Loss')
        plt.xlabel('Iteration')
        if save_flag == 1:
            plt.savefig(path.join(input_path, "loss_cls.pdf"))
            plt.savefig(path.join(input_path, "loss_cls.png"))
        else:
            plt.show()
        plt.clf()

    if path.exists(path.join(input_path, 'loss_loc.txt')):
        x = range(0, len(loc_loss))
        lines = plt.plot(x, loc_loss)
        plt.setp(lines, color='c', linewidth=1.0, alpha=0.5)
        loc_loss_ave = np.convolve(loc_loss, b, mode='same')
        lines = plt.plot(x, loc_loss_ave)
        plt.setp(lines, color='b', linewidth=1.0)
        plt.ylim(-0.1, 1.5)
        plt.ylabel('Localization Loss')
        plt.xlabel('Iteration')
        if save_flag == 1:
            plt.savefig(path.join(input_path, "loss_loc.pdf"))
            plt.savefig(path.join(input_path, "loss_loc.png"))
        else:
            plt.show()
        plt.clf()

    if path.exists(path.join(input_path, 'loss_seg.txt')):
        x = range(0, len(seg_loss))
        lines = plt.plot(x, seg_loss)
        plt.setp(lines, color='c', linewidth=1.0, alpha=0.5)
        seg_loss_ave = np.convolve(seg_loss, b, mode='same')
        lines = plt.plot(x, seg_loss_ave)
        plt.setp(lines, color='b', linewidth=1.0)
        plt.ylim(-0.1, 5)
        plt.ylabel('Segmentation Loss')
        plt.xlabel('Iteration')
        if save_flag == 1:
            plt.savefig(path.join(input_path, "loss_seg.pdf"))
            plt.savefig(path.join(input_path, "loss_seg.png"))
        else:
            plt.show()
        plt.clf()

    plt.close()

if __name__ == '__main__':
    main()

