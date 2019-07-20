import cv2 as cv
import numpy as np

a = np.array([[[0.1, 0.2], [0.3, 0.4]],
            [[0.2, 0.1], [0.4, 0.3]],
            [[0.4, 0.5], [0.7, 0.6]]], np.float32)
print(np.max(a, 0.3))
# b = np.array([[1, 2], [3, 4]], np.float32)
# kernel = np.ones((2, 2), np.uint8)
# a = cv.erode(a, kernel)
# print(a)
# b = cv.erode(b, kernel)
# print(b)
# l = np.array([1, 2, 3])
# print(l < 2)