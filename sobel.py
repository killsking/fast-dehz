import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('img')
parser.add_argument('out', nargs='?')
args = parser.parse_args()

im = cv.imread(args.img, 0)
# im = cv.GaussianBlur(im, (5,5), 0)

sobelx = cv.Sobel(im, cv.CV_64F, 1, 0, ksize=3)
sobely = cv.Sobel(im, cv.CV_64F, 0, 1, ksize=3)
cv.imshow('grad_x', sobelx)
cv.imshow('grad_y', sobely)
cv.waitKey(0)
if args.out is not None:
    cv.imwrite(args.out, sobelx * 255)