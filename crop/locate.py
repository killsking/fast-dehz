import numpy as np
import cv2 as cv
import math

# import argparse

from fpcp import fast_pcp, shrink, flatten, restore, hard_threshold
# from mini_dehz import enhance_lut

# parser = argparse.ArgumentParser()
# parser.add_argument('input')
# parser.add_argument('output')
# args = parser.parse_args()
in_path = 'input/videos/quick.mov'
out_path = 'output/motion/crop.mov'

# cropped shape
h, w = 150, 150

cap = cv.VideoCapture(in_path)
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter(out_path, fourcc, 30.0, (h, w))

ret, frame = cap.read()
assert(ret)
fh, fw = frame.shape
frame = cv.normalize(frame.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
_, frame = cv.threshold(frame, 0.5, 1.0, cv.THRESH_BINARY)

frames = []
count = 0
c0, c1 = 0, 0 # center
d0, d1 = h // 2, w // 2

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break

    # store the first 10 frames for localization
    if count < 10:
        frames.append(frame)
        count += 1
        continue

    # fast pcp
    if count == 10:
        oframes = frames
        for f in frames:
            f = cv.cvtColor(f, cv.COLOR_BGR2GRAY)
        frames = np.dstack(frames)

        M, m, n = flatten(frames)
        lamb = 1 / math.sqrt(max(M.shape))
        L, S = fast_pcp(M, lamb)

        O = hard_threshold(S)
        O_3d = restore(S, m, n)

        V = np.sum(O_3d, axis=2)

        max_score = 0
        for i in range(d0, fh-d0+1):
            for j in range(d1, fw-d1+1):
                if V[i, j] == 0:
                    continue
                score = np.sum(V[i-d0:i+d0, j-d1:j+d1])
                if score > max_score:
                    c0, c1 = i, j
                    max_score = score
        
        for f in oframes:
            # crop
            out.write(f[c0-d0:c0+d0, c1-d1:c1+d1])
        
    out.write(frame[c0-d0:c0+d0, c1-d1:c1+d1])
    count += 1

cap.release()
out.release()

# sum_filter = np.ones((h, w), dtype=np.float32)
# score = convolve2d(V, sum_filter, 'valid')
# # print(score)
# location = np.unravel_index(score.argmax(), score.shape)


top_left = (c1 - d1, c0 - d0)
bottem_right = (c1 + d1, c0 + d0)
cv.rectangle(frame0, top_left, bottem_right, (0,255,0), 3)
cv.imshow('initial_frame', frame0)
cv.waitKey(0)