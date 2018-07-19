#coding: utf-8
import cv2 as cv
import sys
import math
import numpy as np
import random

import common_params


def lookupColorTable(color_steps):

    class_color = []

    for i in xrange(0, len(color_steps)):

        if color_steps[i] < 63:
            b = 255
        elif color_steps[i] < 127:
            b = -4 * color_steps[i] + 510
        else:
            b = 0

        if color_steps[i] < 63:
            g = 4 * color_steps[i]
        elif color_steps[i] < 191:
            g = 255
        else:
            g = -4 * color_steps[i] + 1020

        if color_steps[i] < 127:
            r = 0
        elif color_steps[i] < 191:
            r = 4 * color_steps[i] - 510
        else:
            r = 255

        class_color.append([int(b + 0.5), int(g + 0.5), int(r + 0.5)])

    return class_color

def BGR_2_HSV(img):

    img_r = img.copy()

    n = img.shape[0] * img.shape[1]

    for k in xrange(0, n):
        x = int(k % img.shape[1])
        y = int(k / img.shape[1])
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        max_v = max(max(b, g), r)
        min_v = min(min(b, g), r)
        delta = max_v - min_v
        v = max_v
        if max_v == 0.0:
            s = h = 0.0
        else:
            s = delta / max_v
            if r == max_v:
                h = (g - b) / (delta + 10e-8)
            elif g == max_v:
                h = 2. + (b - r) / (delta + 10e-8)
            else:
                h = 4. + (r - g) / (delta + 10e-8)

            h = h * (np.pi / 3.)

            if h < 0.0:
                h += (2. * np.pi)

            h = h / (2. * np.pi)

            img_r[y, x, 0] = h
            img_r[y, x, 1] = s
            img_r[y, x, 2] = v

    return img_r

def HSV_2_BGR(img):

    img_r = img.copy()

    n = img.shape[0] * img.shape[1]

    for k in xrange(0, n):
        x = int(k % img.shape[1])
        y = int(k / img.shape[1])
        h = img[y, x, 0]
        s = img[y, x, 1]
        v = img[y, x, 2]
        h = h * (2. * np.pi)
        if s == 0.0:
            r = g = b = v
        else:
            idx = int(np.floor(h))
            f = h - idx
            p = v * (1. - s)
            q = v * (1. - s * ((3. / np.pi) * f))
            t = v * (1. - s * (1. - ((3. / np.pi) * f)))
            if idx == 0:
                r = v
                g = t
                b = p
            elif idx == 1:
                r = q
                g = v
                b = p
            elif idx == 2:
                r = p
                g = v
                b = t
            elif idx == 3:
                r = p
                g = q
                b = v
            elif idx == 4:
                r = t
                g = p
                b = v
            else:
                r = v
                g = p
                b = q

        img_r[y, x, 0] = b
        img_r[y, x, 1] = g
        img_r[y, x, 2] = r

    return img_r


def BGR_2_HSV_(img):

    h = np.zeros((img.shape[0], img.shape[1]), np.float32)
    s = np.zeros((img.shape[0], img.shape[1]), np.float32)
    v = np.zeros((img.shape[0], img.shape[1]), np.float32)

    img_r = img.copy()
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    max_v = cv.max(cv.max(b, g), r)
    min_v = cv.min(cv.min(b, g), r)
    delta = max_v - min_v
    v = max_v
    zero_m = (max_v == 0.)
    zero_m = zero_m.astype(np.float32)
    nonzeor_m = (max_v != 0.)
    nonzeor_m = nonzeor_m.astype(np.float32)

    exp_m = zero_m * 10e-8

    s = delta / (max_v + exp_m)
    s = s * nonzeor_m

    rmax = (r == max_v)
    rmax = rmax.astype(np.float32)

    gmax = (g == max_v)
    gr = (g != r)
    gmax = gmax.astype(np.float32)
    gr = gr.astype(np.float32)
    gmax = gmax * gr

    bmax = (b == max_v)
    br = (b != r)
    bg = (b != g)
    bmax = bmax.astype(np.float32)
    br = br.astype(np.float32)
    bg = bg.astype(np.float32)
    bmax = bmax * br * bg

    h += ((g - b) / (delta + 10e-8)) * rmax
    h += (((b - r) / (delta + 10e-8)) + 2.) * gmax
    h += (((r - g) / (delta + 10e-8)) + 4.) * bmax

    h = h * (np.pi / 3.)

    neg_m = (h < 0.0)
    neg_m = neg_m.astype(np.float32)
    h += neg_m * (2. * np.pi)

    h = h / (2. * np.pi)

    img_r[:, :, 0] = h
    img_r[:, :, 1] = s
    img_r[:, :, 2] = v

    return img_r


def HSV_2_BGR_(img):

    r = np.zeros((img.shape[0], img.shape[1]), np.float32)
    g = np.zeros((img.shape[0], img.shape[1]), np.float32)
    b = np.zeros((img.shape[0], img.shape[1]), np.float32)

    img_r = img.copy()
    h = img[:, :, 0]
    s = img[:, :, 1]
    v = img[:, :, 2]

    h = h * (2. * np.pi)

    zero_m = (s == 0.)
    zero_m = zero_m.astype(np.float32)
    nonzeor_m = (s != 0.)
    nonzeor_m = nonzeor_m.astype(np.float32)

    idx = np.floor(h)
    idx = idx.astype(np.int16)

    f = h - idx
    p = v * (1. - s)
    q = v * (1. - s * ((3. / np.pi) * f))
    t = v * (1. - s * (1. - ((3. / np.pi) * f)))

    idx0 = (idx == 0)
    idx0 = idx0.astype(np.float32)

    idx1 = (idx == 1)
    idx1 = idx1.astype(np.float32)

    idx2 = (idx == 2)
    idx2 = idx2.astype(np.float32)

    idx3 = (idx == 3)
    idx3 = idx3.astype(np.float32)

    idx4 = (idx == 4)
    idx4 = idx4.astype(np.float32)

    idxE = idx0 + idx1 + idx2 + idx3 + idx4
    idxE = (idxE == 0)
    idxE = idxE.astype(np.float32)

    r += v * idx0
    g += t * idx0
    b += p * idx0

    r += q * idx1
    g += v * idx1
    b += p * idx1

    r += p * idx2
    g += v * idx2
    b += t * idx2

    r += p * idx3
    g += q * idx3
    b += v * idx3

    r += t * idx4
    g += p * idx4
    b += v * idx4

    r += v * idxE
    g += p * idxE
    b += q * idxE

    r = r * nonzeor_m
    g = g * nonzeor_m
    b = b * nonzeor_m

    r += v * zero_m
    g += v * zero_m
    b += v * zero_m

    img_r[:, :, 0] = b
    img_r[:, :, 1] = g
    img_r[:, :, 2] = r

    return img_r


def distortImage(img, dhue, dsat, dexp):

    img = BGR_2_HSV_(img)

    n = img.shape[0] * img.shape[1]

    img[:, :, 0] = img[:, :, 0] + dhue
    img[:, :, 1] = img[:, :, 1] * dsat
    img[:, :, 2] = img[:, :, 2] * dexp

    m = img[:, :, 0] > 1.
    m = m.astype(np.float32)
    p = img[:, :, 0] < 0.
    p = p.astype(np.float32)

    img[:, :, 0] = img[:, :, 0] - m
    img[:, :, 0] = img[:, :, 0] + p

    img = HSV_2_BGR_(img)

    img = np.minimum(np.maximum(img, 0.), 1.)

    return img

def correct_boxes(box, dx, dy, sx, sy, flip_type):

    left = box[0]
    top = box[1]
    right = box[2]
    bottom = box[3]

    left = left * sx - dx
    right = right * sx - dx
    top = top * sy - dy
    bottom = bottom * sy - dy

    if flip_type == 0:
        swap = top
        top = 1.0 - bottom
        bottom = 1.0 - swap
    elif flip_type == 1:
        swap = left
        left = 1.0 - right
        right = 1.0 - swap

    xmin = min(max(left, 0.), 1.)
    ymin = min(max(top, 0.), 1.)
    xmax = min(max(right, 0.), 1.)
    ymax = min(max(bottom, 0.), 1.)

    return [xmin, ymin, xmax, ymax]

def augmentation(img, idx, gt_boxes):


    oh = img.shape[0]
    ow = img.shape[1]

    dw = int(ow * common_params.jitter + 0.5)
    dh = int(oh * common_params.jitter + 0.5)

    pleft = int(random.uniform(-dw, dw) + 0.5)
    pright = int(random.uniform(-dw, dw) + 0.5)
    ptop = int(random.uniform(-dh, dh) + 0.5)
    pbot = int(random.uniform(-dh, dh) + 0.5)

    swidth = ow - pleft - pright
    sheight = oh - ptop - pbot

    sx = float(swidth) / float(ow)
    sy = float(sheight) / float(oh)

    if pleft < 0:
        xmin_pad = -1 * pleft
        px = 0
    else:
        xmin_pad = 0
        px = pleft

    if pright < 0:
        xmax_pad = -1 * pright
    else:
        xmax_pad = 0

    if ptop < 0:
        ymin_pad = -1 * ptop
        py = 0
    else:
        ymin_pad = 0
        py = ptop

    if pbot < 0:
        ymax_pad = -1 * pbot
    else:
        ymax_pad = 0

    dhue = random.uniform(-1. * common_params.hue, common_params.hue)
    dsat = random.uniform(1., common_params.saturation)
    dexp = random.uniform(1., common_params.exposure)

    if random.randint(0, 1) == 0:
        dsat = 1. / dsat

    if random.randint(0, 1) == 0:
        dexp = 1. / dexp

    hsv_param = [999., 999., 999.]

    if random.randint(0, 4) >= 1:
        img = img.astype(np.float32)
        img /= 255.
        img = distortImage(img, dhue, dsat, dexp)
        img *= 255.
        img = img.astype(np.uint8)
        hsv_param[0] = dhue
        hsv_param[1] = dsat
        hsv_param[2] = dexp

    if common_params.border_type == cv.BORDER_CONSTANT:
        cropped_img = cv.copyMakeBorder(img, ymin_pad, ymax_pad, xmin_pad, xmax_pad, common_params.border_type, value = common_params.border_val)
    else:
        cropped_img = cv.copyMakeBorder(img, ymin_pad, ymax_pad, xmin_pad, xmax_pad, common_params.border_type)

    cropped_img = cv.getRectSubPix(cropped_img, (swidth, sheight), (px + (swidth / 2.), py + (sheight / 2.)))
    cropped_img = cv.resize(cropped_img, (common_params.insize, common_params.insize), interpolation = cv.INTER_CUBIC)

    dx = (float(pleft) / float(ow)) / sx
    dy = (float(ptop) / float(oh)) / sy

    border_pixels = [xmin_pad, ymin_pad, xmax_pad, ymax_pad]
    crop_param = [px + (swidth / 2.), py + (sheight / 2.), swidth, sheight]

    flip_type = random.randint(-1, 1)

    if flip_type >= 0:
        cropped_img = cv.flip(cropped_img, flip_type)

    arg_boxes = []
    arg_idx = []

    for i in xrange(0, len(gt_boxes)):

        gt_xmin = gt_boxes[i][0] / common_params.insize
        gt_ymin = gt_boxes[i][1] / common_params.insize
        gt_xmax = gt_boxes[i][2] / common_params.insize
        gt_ymax = gt_boxes[i][3] / common_params.insize

        new_box = correct_boxes([gt_xmin, gt_ymin, gt_xmax, gt_ymax], dx, dy, 1.0 / sx, 1.0 / sy, flip_type)

        gt_xmin = new_box[0] * common_params.insize
        gt_ymin = new_box[1] * common_params.insize
        gt_xmax = new_box[2] * common_params.insize
        gt_ymax = new_box[3] * common_params.insize

        width = gt_xmax - gt_xmin
        height = gt_ymax - gt_ymin

        if ((width >= 10.) and (height >= 10.)):
            arg_boxes.append([gt_xmin, gt_ymin, gt_xmax, gt_ymax])
            arg_idx.append(idx[i])

    return (cropped_img, arg_idx, arg_boxes, border_pixels, crop_param, hsv_param, flip_type)

def trainAugmentation(img, border_pixels, crop_param, hsv_param, flip_type, segimg):
    hsv_srand = 0

    if hsv_srand == 1:
        dhue = hsv_param[0]
        dsat = hsv_param[1]
        dexp = hsv_param[2]
    else:
        dhue = random.uniform(-1. * common_params.hue, common_params.hue)
        dsat = random.uniform(1., common_params.saturation)
        dexp = random.uniform(1., common_params.exposure)
        if random.randint(0, 1) == 0:
            dsat = 1. / dsat
        if random.randint(0, 1) == 0:
            dexp = 1. / dexp

    if random.randint(0, 4) >= 1:
        img = img.astype(np.float32)
        img /= 255.
        img = distortImage(img, dhue, dsat, dexp)
        img *= 255.
        img = img.astype(np.uint8)


    if common_params.border_type == cv.BORDER_CONSTANT:
        cropped_img = cv.copyMakeBorder(img, border_pixels[1], border_pixels[3], border_pixels[0], border_pixels[2], common_params.border_type, value = common_params.border_val)
        cropped_segimg = cv.copyMakeBorder(segimg, border_pixels[1], border_pixels[3], border_pixels[0], border_pixels[2], cv.BORDER_REPLICATE)
    else:
        cropped_img = cv.copyMakeBorder(img, border_pixels[1], border_pixels[3], border_pixels[0], border_pixels[2], common_params.border_type)
        cropped_segimg = cv.copyMakeBorder(segimg, border_pixels[1], border_pixels[3], border_pixels[0], border_pixels[2], cv.BORDER_REPLICATE)

    cropped_img = cv.getRectSubPix(cropped_img, (int(crop_param[2]), int(crop_param[3])), (crop_param[0], crop_param[1]))
    cropped_img = cv.resize(cropped_img, (common_params.insize, common_params.insize), interpolation = cv.INTER_CUBIC)


    w = crop_param[2]
    h = crop_param[3]
    cx = crop_param[0]
    cy = crop_param[1]
    cropped_segimg = cropped_segimg[int(cy-h/2):int(cy+h/2), int(cx-w/2):int(cx+w/2)]

    cropped_segimg = cv.resize(cropped_segimg, (common_params.insize, common_params.insize), interpolation = cv.INTER_NEAREST)


    if flip_type >= 0:
        cropped_img = cv.flip(cropped_img, flip_type)
        cropped_segimg = cv.flip(cropped_segimg, flip_type)

    return (cropped_img, cropped_segimg)
