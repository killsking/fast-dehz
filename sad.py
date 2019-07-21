import numpy as np
import cv2 as cv
import argparse

from dehz import dehz

parser = argparse.ArgumentParser(description='')
parser.add_argument('video_path')
parser.add_argument('depth_path', nargs='?')
args = parser.parse_args()

cap_v = cv.VideoCapture(args.video_path)
with_depth = args.depth_path is not None
if with_depth:
    cap_d = cv.VideoCapture(args.depth_path)

runtime = 0.0
frame_count = 0
while(cap_v.isOpened()):
    frame = cap_v.read()[1]
    frame_count += 1
    if with_depth:
        depth = cap_d.read()[1]
        depth = cv.cvtColor(depth, cv.COLOR_BGR2GRAY)

        e1 = cv.getTickCount()
        frame = dehz(frame, depth)
        e2 = cv.getTickCount()

    else:
        e1 = cv.getTickCount()
        frame = dehz(frame)
        e2 = cv.getTickCount()

    cv.imshow('frame', frame)
    runtime += (e2 - e1) / cv.getTickFrequency()

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

print('average runtime: %.5fs' % (runtime / frame_count))
cap_v.release()
if with_depth:
    cap_d.release()
cv.destroyAllWindows()