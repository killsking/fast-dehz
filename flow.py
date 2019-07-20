import numpy as np
import cv2 as cv
import argparse

from dehz import dehz

parser = argparse.ArgumentParser(description='')
parser.add_argument('video_path', help='Path to the video')
args = parser.parse_args()

cap = cv.VideoCapture(args.video_path)

while(cap.isOpened()):
    ret, frame = cap.read()
    
    # frame = dehz(frame)
    cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv.destroyAllWindows()