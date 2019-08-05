import cv2 as cv

from gamma import generate_lows
from flow import enhance

# src = '../JUST_A_TEST_'
src = '/mnt/data/yxchen/gesture-datasets/ems/data/subject01_setting3_'
nums = ['08', '09', '10']
for n in nums:
    path = src + n
    folders = generate_lows(path)
    for fdn in folders:
        enhance(fdn)
