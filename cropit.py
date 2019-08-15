import numpy as np
import cv2 as cv

inpath = ''
outpath = ''

cap = cv.VideoCapture(in_path)
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter(out_path, fourcc, 30.0, (h, w))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break

    out.write(frame[235:385, 335:485])

cap.release()
out.release()