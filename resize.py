import numpy as np
import cv2 as cv

cap = cv.VideoCapture('../others/night.mp4')

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'MPEG')
out = cv.VideoWriter('input/night.mp4',fourcc, 30.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv.resize(frame, (640, 480))
        # write the flipped frame
        out.write(frame)
        cv.imshow('frame',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
# cv.destroyAllWindows()