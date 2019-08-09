import numpy as np
import cv2 as cv

def adjust_gamma(image, gamma=1.0):
    # invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** gamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    return cv.LUT(image, table)

def main():
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    gamma = 0.1
    for i in range(1, 10):
        cap = cv.VideoCapture('input/quick.mov')
        out = cv.VideoWriter('input/gamma_' + str(gamma) + '.mov', fourcc, 30.0, (640,480))

        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break

            frame = adjust_gamma(frame, gamma)
            out.write(frame)
            # cv.imshow('frame', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        out.release()
        cap.release()

        gamma += 0.1

if __name__ == "__main__":
    im = cv.imread('0.jpg')
    gamma = 2.5
    # im = adjust_gamma(im, gamma)
    im = im / gamma
    cv.imwrite('g004.jpg', im)
    cv.imshow('bad', im)

    # im = adjust_gamma(im, 1 / gamma)
    im = im * gamma
    cv.imshow('good', im)
    
    cv.waitKey()