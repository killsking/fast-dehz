import numpy as np
import cv2 as cv
import argparse
import heapq

def dehz(im, w = 0.8):
    assert w > 0

    # normalize
    if np.max(im.ravel() > 1):
        im_n = cv.normalize(im.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

    # invert
    R = 1 - im_n
    S = np.sum(R, axis=2)
    num = int(S.size * 0.002) # empirical

    # erode
    kernel = np.ones((7,7), np.uint8)
    R_d = cv.erode(R, kernel)

    # calculate global atmosphere light A
    M = np.min(R_d, axis=2)
    M_s = set(heapq.nlargest(num, M.ravel()))
    maxS = 0
    for index, m in np.ndenumerate(M):
        if m in M_s and S[index] > maxS:
            maxS = S[index]
            A = R_d[index]

    T = 1 - w * np.min(R_d / A, axis=2)
    for row in T:
        for t in row:
            if t < 0.5:
                t *= 2

    # restore
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            R[i, j] = R[i, j] - A
    for k in range(R.shape[2]):
        R[:, :, k] /= T
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            R[i, j] = 1 - (R[i, j] + A)

    return cv.normalize(R, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('image_path', help='Path to the image')
    args = parser.parse_args()

    im = cv.imread(args.image_path)

    e1 = cv.getTickCount()
    im = dehz(im)
    e2 = cv.getTickCount()

    t = (e2 - e1) / cv.getTickFrequency()
    print(t)
    cv.imshow('disp', im)
    cv.waitKey(0)
    cv.imwrite('output.jpg', im)