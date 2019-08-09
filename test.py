import cv2 as cv
import numpy as np
from scipy import signal

a = np.array([[[0.1, 0.2], [0.3, 0.4]],
            [[0.2, 0.1], [0.4, 0.3]],
            [[0.4, 0.5], [0.7, 0.6]]], np.float32)
# print(np.max(a, 0.3))

dist = np.array(
    [[0,1,2,9],
    [3,4,5,9],
    [6,7,8,9],
    [1,2,3,5]]
)
sum_filter = np.array(
    [[1,1],
    [1,1]]
)
block_h = block_w = 2
# dist = cv.absdiff(frame, last_frame)
# sub_mats = np.lib.stride_tricks.as_strided(dist, )
sums = signal.convolve2d(dist, sum_filter, mode='valid')[::block_h, ::block_w]
print(sums)


# b = np.array([[1, 2], [3, 4]], np.float32)
# kernel = np.ones((2, 2), np.uint8)
# a = cv.erode(a, kernel)
# print(a)
# b = cv.erode(b, kernel)
# print(b)
# l = np.array([1, 2, 3])
# print(l < 2)