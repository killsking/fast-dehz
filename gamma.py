import numpy as np
import cv2 as cv

def adjust_gamma(image, gamma=1.0):
   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv.LUT(image, table)

# x = 'input/8.bmp'  #location of the image
# original = cv.imread(x, 1)
# cv.imshow('original', original)

# gamma = 2                                   # change the value here to get different result
# adjusted = adjust_gamma(original, gamma=gamma)
# cv.putText(adjusted, "g={}".format(gamma), (10, 30),cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
# cv.imshow("gammam image 1", adjusted)

# cv.waitKey(0)
# cv.destroyAllWindows()

if __name__ == "__main__":
    cap = cv.VideoCapture('input/quick.mov')
    fourcc = cv.VideoWriter_fourcc(*'MPEG')
    out = cv.VideoWriter('input/dim.mov', fourcc, 30.0, (640,480))

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break

        frame = adjust_gamma(frame, 0.5)
        out.write(frame)
        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
