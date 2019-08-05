import numpy as np
import cv2 as cv

import argparse
from os.path import basename, normpath, exists
from os import makedirs

def adjust_gamma(image, gamma=1.0):
   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv.LUT(image, table)

def generate_lows(fdn):
    folders = []

    # fourcc = cv.VideoWriter_fourcc(*'mp4v')
    fourcc = cv.cv.FOURCC(*'mp4v')
    gamma = 0.1

    for i in range(1, 10):
        cap = cv.VideoCapture(fdn + '/rgb/quick.mov')

        output_dir = './data/' + basename(normpath(fdn)) + '_gamma' + str(i)
        if not exists(output_dir):
            makedirs(output_dir)
        else:
            continue

        folders.append(output_dir)
        print(output_dir)

        out = cv.VideoWriter(output_dir + '/input.mov',
                            fourcc, 30.0, (640,480))
        
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break

            frame = adjust_gamma(frame, gamma)
            out.write(frame)
            # cv.imshow('frame', frame)
            # if cv.waitKey(1) & 0xFF == ord('q'):
            #     break

        out.release()
        cap.release()

        gamma += 0.1

    return folders

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('gamma')
    parser.add_argument('input_fdn') # folder name
    # parser.add_argument('output_fn')
    args = parser.parse_args()

    generate_lows(args.input_fdn)