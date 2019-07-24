import numpy as np
import cv2 as cv
import argparse
import heapq

def dehz(im, depth=None, w=0.8):
    assert w > 0

    # normalize
    # if np.max(im.ravel() > 1):
    im_n = cv.normalize(im.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

    # invert
    R = 1 - im_n
    S = np.sum(R, axis=2)
    num = int(S.size * 0.002) # empirical

    if depth is None:
        # erode to avoid over exposure
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

        A = np.ones(3, dtype=float)

        T = 1 - w * np.min(R_d / A, axis=2)
        # for row in T:
        #     for t in row:
        #         if t < 0.5:
        #             t *= 2
    else:
        depth_map = cv.normalize(depth.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

        kernel = np.ones((7,7), np.uint8)
        depth_map = cv.erode(depth_map, kernel)

        T = 1 - w * depth_map # TODO: w

    # restore
    for k in range(R.shape[2]):
        R[:, :, k] = 1 - ((R[:, :, k] - A[k]) / T + A[k])
        
    # for i in range(R.shape[0]):
    #     for j in range(R.shape[1]):
    #         R[i, j] -= A
    # for k in range(R.shape[2]):
    #     R[:, :, k] /= T
    # for i in range(R.shape[0]):
    #     for j in range(R.shape[1]):
    #         R[i, j] = 1 - (R[i, j] + A)

    return cv.normalize(R, None, 0.0, 255.0, cv.NORM_MINMAX).astype(np.uint8)

# with motion estimation
def dehz_me(im, im_n, T, depth=None, w=0.8):
    assert w > 0
    block_num = 16
    block_h = im.shape[0] // block_num
    block_w = im.shape[1] // block_num
    threshold = block_h * block_w * 3

    # normalize and invert
    R = 1 - cv.normalize(im.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

    # cnt = 0

    if depth is None:
        # erode to avoid over exposure
        kernel = np.ones((7,7), np.uint8)
        R_d = cv.erode(R, kernel)

        # macroblocks
        if T is None:
            T = 1 - w * np.min(R_d, axis=2)
            # cnt += block_num * block_num
        else:
            for i in range(block_num):
                for j in range(block_num):
                    mb = im[i * block_h : (i + 1) * block_h, j * block_w : (j + 1) * block_w]
                    mb_n = im_n[i * block_h : (i + 1) * block_h, j * block_w : (j + 1) * block_w]
                    if np.sum(cv.absdiff(mb, mb_n)) > threshold:
                        # print('recalculated')
                        # cnt += 1
                        mb_r = R_d[i * block_h : (i + 1) * block_h, j * block_w : (j + 1) * block_w]
                        T[i * block_h : (i + 1) * block_h, j * block_w : (j + 1) * block_w] = 1 - w * np.min(mb_r, axis=2)

    else:
        depth_map = cv.normalize(depth.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

        kernel = np.ones((7,7), np.uint8)
        depth_map = cv.erode(depth_map, kernel)

        T = 1 - w * depth_map # TODO: w

    # print(cnt)
    # restore
    for k in range(R.shape[2]):
        R[:, :, k] = 1 - ((R[:, :, k] - 1) / T + 1)

    return cv.normalize(R, None, 0.0, 255.0, cv.NORM_MINMAX).astype(np.uint8), T


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('image_path')
    parser.add_argument('depth_path', nargs='?')
    args = parser.parse_args()

    im = cv.imread(args.image_path)
    with_depth = args.depth_path is not None
    if with_depth:
        d = cv.imread(args.depth_path, 0)

    if with_depth:
        e1 = cv.getTickCount()
        im1 = dehz(im, d)
        e2 = cv.getTickCount()

        t = (e2 - e1) / cv.getTickFrequency()
        print(t) 

        cv.imshow('withdepth', im1)
        
    else:
        e1 = cv.getTickCount()
        im2 = dehz(im)
        e2 = cv.getTickCount()

        t = (e2 - e1) / cv.getTickFrequency()
        print(t) 

        cv.imshow('withoutdepth', im2)

    cv.waitKey(0)
    # cv.imwrite('output.jpg', im)