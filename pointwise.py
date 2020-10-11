import numpy as np
import cv2 as cv
import math
import argparse

def op_dehz(x, w=0.8):
    return x / (1 - w + w * x)

def op_gamma(x, gamma=1.0):
    return x ** gamma

def op_log(x, base = 10):
    return math.log(x + 1, base)

def enhance(im):
    h, w, c = im.shape
    im_n = cv.normalize(im.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
    for k in range(c):
        for i in range(h):
            for j in range(w):
                im_n[i,j,k] = op_dehz(im_n[i,j,k])
                # im_n[i,j,k] = op_gamma(im_n[i,j,k], 0.4)
                # im_n[i,j,k] = op_log(im_n[i,j,k])
    im = cv.normalize(im_n, None, 0.0, 255.0, cv.NORM_MINMAX).astype(np.uint8)
    return im

def enhance_lut(im):
    tb = np.array([op_dehz(i / 255) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    return cv.LUT(im, tb)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()
    im = cv.imread(args.path)

    e1 = cv.getTickCount()
    # im = enhance(im)
    im = enhance_lut(im)
    e2 = cv.getTickCount()
    runtime = (e2 - e1) / cv.getTickFrequency()

    print(runtime)
    cv.imshow('done', im)
    cv.waitKey(0)

