import cv2 as cv
import argparse
parser = argparse.ArgumentParser(description='Code for Histogram Equalization tutorial.')
parser.add_argument('--input', help='Path to input image.')
args = parser.parse_args()

src = cv.imread(args.input)
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)
    
e1 = cv.getTickCount()
# src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
img_yuv = cv.cvtColor(src, cv.COLOR_BGR2YUV)

# equalize the histogram of the Y channel
img_yuv[:,:,0] = cv.equalizeHist(img_yuv[:,:,0])

# convert the YUV image back to RGB format
img_output = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)
e2 = cv.getTickCount()

runtime = (e2 - e1) / cv.getTickFrequency()
print(runtime)

cv.imshow('Color input image', src)
cv.imshow('Histogram equalized', img_output)

cv.waitKey(0)
