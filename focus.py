import numpy as np
import cv2 as cv

from fpcp import fast_pcp, shrink, flatten, restore, hard_threshold
from mini_dehz import enhance_lut

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('fn')
    args = parser.parse_args()

    cap = cv.VideoCapture(args.fn)

    # get first 100 frames
    V = []
    # while (cap.isOpened()):
    for i in range(100):
        ret, frame = cap.read(0)
        if ret == False:
            break
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        V.append(frame)
    V = np.dstack(V)

    M, m, n = flatten(V)
    lamb = 1 / math.sqrt(max(M.shape))
    L, S, ranks, rhos = fast_pcp(M, lamb)

    # outlier
    O = hard_threshold(S)

    L_3d = restore(L, m, n)
    S_3d = restore(S, m, n)
    O_3d = restore(O, m, n)