import numpy as np
import cv2 as cv
import rawpy

im = cv.imread('input/8.bmp')
b, g, r = im[:, :, 0], im[:, :, 1], im[:, :, 2]

output = np.zeros((im.shape[0] * 2, im.shape[1] * 2))
output[::2, ::2] = r
output[::2, 1::2] = g
output[1::2, ::2] = g
output[1::2, 1::2] = b

cv.imshow('raw', output)
cv.waitKey(0)
# rawpy.imsave('8.raw', output)