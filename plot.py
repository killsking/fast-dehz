import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from glob import glob
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('gt_path')
# parser.add_argument('output_path')
# args = parser.parse_args()

def get_psnr(gt_path, output_path):
    cap_gt = cv.VideoCapture(gt_path)
    cap_o = cv.VideoCapture(output_path)

    psnr_list = []
    while (cap_gt.isOpened() and cap_o.isOpened()):
        ret_gt, frame_gt = cap_gt.read()
        ret_o, frame_o = cap_o.read()
        if ret_gt == False or ret_o == False:
            break
        
        psnr_list.append(cv.PSNR(frame_gt, frame_o))

    return psnr_list


def plot_psnr2frame(gt_path, output_path):
    psnr_list = get_psnr(gt_path, output_path)

    x = np.arange(1, len(psnr_list) + 1)
    plt.plot(x, psnr_list)
    plt.show()


def plot_psnr2illum(gt_path, output_paths, label):
    avg_psnr_list = []
    illum_cnt = 0
    for output_path in output_paths:
        psnr_list = get_psnr(gt_path, output_path)
        avg_psnr = np.sum(psnr_list) / len(psnr_list)

        avg_psnr_list.append(avg_psnr)
        illum_cnt += 1

    x = np.arange(0.1, 1.0, 0.1)
    plt.plot(x, avg_psnr_list, label=label)
    print(avg_psnr_list)
    # plt.show()

gt_path = '../EMS_code/data/subject01_setting3_08_gt/quick.mov'
folder_paths = glob('../EMS_code/data/subject01_setting3_08_gamma?')
output_paths = [x + '/output.mov' for x in folder_paths]
input_paths = [x + '/input.mov' for x in folder_paths]

plot_psnr2illum(gt_path, input_paths, 'input')
plot_psnr2illum(gt_path, output_paths, 'output')
plt.xlabel('Gamma')
plt.ylabel('PSNR (dB)')
plt.grid()
plt.legend(loc='best')
plt.show()