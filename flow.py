import numpy as np
import cv2 as cv
import argparse

from dehz import dehz, fdehz
# from gamma import adjust_gamma

parser = argparse.ArgumentParser()
parser.add_argument('path')
parser.add_argument('--fast', action='store_true')
args = parser.parse_args()

cap_v = cv.VideoCapture(args.path)
f = fdehz if args.fast else dehz

runtime = 0.0
frame_count = 0
while(cap_v.isOpened()):
    ret, frame = cap_v.read()
    if ret == False:
        break

    frame_count += 1

    e1 = cv.getTickCount()
    frame = f(frame)
    e2 = cv.getTickCount()

    cv.imshow('frame', frame)
    runtime += (e2 - e1) / cv.getTickFrequency()

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

print('average runtime: %.5fs' % (runtime / frame_count))
cap_v.release()
cv.destroyAllWindows()
