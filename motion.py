import numpy as np
import cv2 as cv
import argparse

# from scipy import signal
# from numpy.fft import fft2, ifft2

from dehz import dehz_me

# def fft_convolve2d(x,y):
#     """
#     2D convolution, using FFT
#     """
#     fr = fft2(x)
#     fr2 = fft2(np.flipud(np.fliplr(y)))
#     m,n = fr.shape
#     cc = np.real(ifft2(fr * fr2))
#     cc = np.roll(cc, -m / 2 + 1, axis=0)
#     cc = np.roll(cc, -n / 2 + 1, axis=1)
#     return cc

if __name__ == "__main__":
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

    # parameters
    w = 0.8
    div = 16
    gop = 30
    print('gop:', gop)

    block_h = 480 // div
    block_w = 640 // div
    threshold = block_h * block_w * 6
    sum_filter = np.ones((block_h, block_w), dtype=np.float32)

    while(cap_v.isOpened()):
        ret, frame = cap_v.read()
        if ret == False:
            break

        # equalize the histogram of the Y channel
        # img_yuv = cv.cvtColor(frame, cv.COLOR_BGR2YUV)
        # img_yuv[:,:,0] = cv.equalizeHist(img_yuv[:,:,0])
        # frame = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)

        if with_depth:
            depth = cap_d.read()[1]
            depth = cv.cvtColor(depth, cv.COLOR_BGR2GRAY)

            e1 = cv.getTickCount()
            new_frame, T = dehz_me(frame, last_frame, T, depth)
            e2 = cv.getTickCount()

        else:
            e1 = cv.getTickCount()
        
            # normalize and invert
            L_inv = 1 - cv.normalize(frame.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
            L_inv[0:480, 0:250] = 0
            L_inv[0:200, 0:640] = 0
        
            kernel = np.ones((7,7), np.uint8)
            L_inv = cv.erode(L_inv, kernel)
    
            # macroblocks
            if frame_count % gop == 0: # recalculation
                T = 1 - w * np.min(L_inv, axis=2)
            else:
                # # calculate l_1 distance
                dist = cv.absdiff(frame, last_frame)
                # # sub_mats = np.lib.stride_tricks.as_strided(dist, )

                # # get the indices of over-threshold sums
                # res = np.where(sums > threshold)
                # active_blocks = list(zip(res[0], res[1]))

                # print(active_blocks)
                # for i, j in active_blocks:
                #     T[i * block_h : (i + 1) * block_h, j * block_w: (j + 1) * block_w] = 1
                
                for i in range(div):
                    for j in range(div):
                        # blk = (i * block_h : (i + 1) * block_h, j * block_w : (j + 1) * block_w)
                        # mb = last_frame[i * block_h : (i + 1) * block_h, j * block_w : (j + 1) * block_w]
                        # mb_n = frame[i * block_h : (i + 1) * block_h, j * block_w : (j + 1) * block_w]
                        # if np.sum(cv.absdiff(mb, mb_n)) > threshold:
                        if np.sum(dist[i * block_h : (i + 1) * block_h, j * block_w : (j + 1) * block_w]) > threshold:
                            # print(i, j)
                            if np.sum(frame[i * block_h : (i + 1) * block_h, j * block_w : (j + 1) * block_w]) < block_h * block_w * 100:
                                T[i * block_h : (i + 1) * block_h, j * block_w : (j + 1) * block_w] = 0.2
                            # mb_r = R_d[i * block_h : (i + 1) * block_h, j * block_w : (j + 1) * block_w]
                            # T[i * block_h : (i + 1) * block_h, j * block_w : (j + 1) * block_w] = 1 - w * np.min(mb_r, axis=2)
        
        
            # restore
            R = np.zeros(frame.shape, dtype=np.float32)
            for k in range(3):
                R[:, :, k] = frame[:, :, k] / T
        
            new_frame = cv.normalize(R, None, 0.0, 255.0, cv.NORM_MINMAX).astype(np.uint8)

            e2 = cv.getTickCount()

        last_frame = frame
        frame_count += 1
        cv.imshow('frame', new_frame)
        runtime += (e2 - e1) / cv.getTickFrequency()

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    print('average runtime: %.5fs' % (runtime / frame_count))
    cap_v.release()
    if with_depth:
        cap_d.release()
    cv.destroyAllWindows()

