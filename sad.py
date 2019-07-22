import numpy as np
import cv2 as cv
import argparse

from dehz import dehz_me

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
last_frame = None
T = None
while(cap_v.isOpened()):
    ret, frame = cap_v.read()
    if ret == False:
        break

    frame_count += 1
    if with_depth:
        depth = cap_d.read()[1]
        depth = cv.cvtColor(depth, cv.COLOR_BGR2GRAY)

        e1 = cv.getTickCount()
        new_frame, T = dehz_me(frame, last_frame, T, depth)
        e2 = cv.getTickCount()

    else:
        e1 = cv.getTickCount()
        new_frame, T = dehz_me(frame, last_frame, T)
        e2 = cv.getTickCount()

    last_frame = frame
    cv.imshow('frame', new_frame)
    runtime += (e2 - e1) / cv.getTickFrequency()

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

print('average runtime: %.5fs' % (runtime / frame_count))
cap_v.release()
if with_depth:
    cap_d.release()
cv.destroyAllWindows()