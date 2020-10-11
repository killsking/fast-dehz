"""DeHz and Fast DeHz"""

import argparse
import heapq
import numpy as np
import cv2 as cv


def dehz(im, w=0.8):
    assert w > 0

    # normalize
    L = cv.normalize(im.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

    # invert
    L_inv = 1 - L
    S = np.sum(L_inv, axis=2)
    num = int(S.size * 0.002) # empirical

    # calculate global atmosphere light A
    M = np.min(L_inv, axis=2)
    M_s = set(heapq.nlargest(num, M.ravel()))
    maxS = 0
    for index, m in np.ndenumerate(M):
        if m in M_s and S[index] > maxS:
            maxS = S[index]
            A = L_inv[index]

    A = np.ones(3, dtype=float)
    T = 1 - w * L_inv[:, :, 2]

    # restore
    for k in range(3):
        L_inv[:, :, k] = 1 - ((L_inv[:, :, k] - A[k]) / T + A[k])

    return cv.normalize(L_inv, None, 0.0, 255.0, cv.NORM_MINMAX).astype(np.uint8)


def fdehz(im, w=0.8):
    assert w > 0

    # normalize
    # if np.max(im.ravel() > 1):
    L = cv.normalize(im.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

    # invert
    L_inv = 1 - L
    # S = np.sum(R, axis=2)
    # num = int(S.size * 0.002) # empirical

    # kernel = np.ones((7,7), np.uint8)
    # L_inv = cv.erode(L_inv, kernel)

    # calculate global atmosphere light A
    # M = np.min(L_inv, axis=2)
    # M_s = set(heapq.nlargest(num, M.ravel()))
    # maxS = 0
    # for index, m in np.ndenumerate(M):
    #     if m in M_s and S[index] > maxS:
    #         maxS = S[index]
    #         A = L_inv[index]
            # coord = index

    # A = np.ones(3, dtype=float)
    # print(A, coord)
    T = 1 - w * np.min(L_inv, axis=2)
    # T = 1 - w * L_inv[:, :, 2]
    # for row in T:
    #     for t in row:
    #         if t < 0.5:
    #             t *= 2

    # Tp = cv.normalize(T, None, 0.0, 255.0, cv.NORM_MINMAX).astype(np.uint8)

    # restore
    R = np.zeros(L.shape, dtype=np.float32)
    for k in range(3):
        # R[:, :, k] = 1 - ((R[:, :, k] - A[k]) / T + A[k])
        R[:, :, k] = L[:, :, k] / T

    return cv.normalize(R, None, 0.0, 255.0, cv.NORM_MINMAX).astype(np.uint8)


# with motion estimation
def dehz_me(im, im_n, T, w=0.8):
    assert w > 0

    div = 16
    # gop = 30

    block_h = im.shape[0] // div
    block_w = im.shape[1] // div
    threshold = block_h * block_w * 3 # parameter

    # normalize and invert
    L_inv = 1 - cv.normalize(im.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

    # cnt = 0

    kernel = np.ones((7,7), np.uint8)
    L_inv = cv.erode(L_inv, kernel)

    # macroblocks
    # if cnt % gop == 0:
    if T is None:
        T = 1 - w * np.min(L_inv, axis=2)
        # cnt += div * div
    else:
        for i in range(div):
            for j in range(div):
                mb = im[i * block_h : (i + 1) * block_h, j * block_w : (j + 1) * block_w]
                mb_n = im_n[i * block_h : (i + 1) * block_h, j * block_w : (j + 1) * block_w]
                if np.sum(cv.absdiff(mb, mb_n)) > threshold:
                    # cnt += 1
                    mb_r = L_inv[i * block_h : (i + 1) * block_h, j * block_w : (j + 1) * block_w]
                    T[i * block_h : (i + 1) * block_h, j * block_w : (j + 1) * block_w] = 1 - w * np.min(mb_r, axis=2)

    # print(cnt)
    # restore
    for k in range(R.shape[2]):
        R[:, :, k] = 1 - ((R[:, :, k] - 1) / T + 1)

    return cv.normalize(R, None, 0.0, 255.0, cv.NORM_MINMAX).astype(np.uint8), T


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--fast', action='store_true')
    args = parser.parse_args()

    im_ori = cv.imread(args.path)
    f = fdehz if args.fast else dehz

    e1 = cv.getTickCount()
    im_enh = f(im_ori)
    e2 = cv.getTickCount()

    t = (e2 - e1) / cv.getTickFrequency()
    print(f'time: {t}')

    # cv.imwrite('output.bmp', im1)
    cv.imshow('withdepth', im_enh)
        
    cv.waitKey(0)
