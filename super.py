import numpy as np
import cv2 as cv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('image_path')
args = parser.parse_args()

im = cv.imread(args.image_path)

hsv = cv.cvtColor(im, cv.COLOR_BGR2HSV)
hsv[:,:,2] = hsv[:,:,2] * 0.1
out = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

# plt.subplot(111), plt.imshow(cv.cvtColor(hsvImg,cv.COLOR_HSV2RGB))
# plt.title('brightened image'), plt.xticks([]), plt.yticks([])
# plt.show()

cv.imwrite('dim.bmp', out)
cv.imshow('dim', out)
cv.waitKey(0)