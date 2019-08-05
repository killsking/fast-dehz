import numpy as np
import cv2 as cv

import argparse
import os

from dehz import dehz

def enhance(fdn, with_depth=False):
    cap_v = cv.VideoCapture(fdn + '/input.mov')
    if with_depth:
        cap_d = cv.VideoCapture(fdn + '/depth.mov')

    # fourcc = cv.VideoWriter_fourcc(*'mp4v')
    fourcc = cv.cv.FOURCC(*'mp4v')
    out = cv.VideoWriter(fdn + '/output.mov', fourcc, 30.0, (640,480))

    runtime = 0.0
    frame_count = 0
    while(cap_v.isOpened()):
        ret, frame = cap_v.read()
        if ret == False:
            break

        frame_count += 1
        if with_depth:
            depth = cap_d.read()[1]
            depth = cv.cvtColor(depth, cv.COLOR_BGR2GRAY)

            e1 = cv.getTickCount()
            frame = dehz(frame, depth)
            e2 = cv.getTickCount()

        else:
            e1 = cv.getTickCount()
            frame = dehz(frame)
            e2 = cv.getTickCount()

        # cv.imshow('frame', frame)
        out.write(frame)
        runtime += (e2 - e1) / cv.getTickFrequency()

        # if cv.waitKey(1) & 0xFF == ord('q'):
        #     break

    print('average runtime: %.5fs' % (runtime / frame_count))
    out.release()
    cap_v.release()
    if with_depth:
        cap_d.release()
    # cv.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input_fdn') # folder name
    parser.add_argument('depth_fdn', nargs='?')
    # parser.add_argument('output_path')
    args = parser.parse_args()

    # with_depth = args.depth_fn is not None
    enhance(args.input_fdn)