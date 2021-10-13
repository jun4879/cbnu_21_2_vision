import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('data/image_Peppers512rgb.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, -1, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cv2.namedWindow('Peppers')
finish = False
while not finish:
    cv2.imshow("Peppers", img)
    key = cv2.waitKey(1)

    if key == ord('e'):
        eroded_iteration = int(input('eroded_iteration: '))
        eroded = cv2.morphologyEx(binary, cv2.MORPH_ERODE, (3, 3), iterations=eroded_iteration)
        #cv2.imshow('eroded / iteration : {0}'.format(eroded_iteration), eroded)
        plt.imshow(eroded, cmap='gray')
        plt.show()

    if key == ord('d'):
        dilated_iteration = int(input('dilated_iteration: '))
        dilated = cv2.morphologyEx(binary, cv2.MORPH_DILATE, (3,3), iterations=dilated_iteration)
        #cv2.imshow('dilated / iteration : {0}'.format(dilated_iteration), dilated)
        plt.imshow(dilated, cmap='gray')
        plt.show()

    if key == ord('o'):
        opened_iteration = int(input('open_iteration: '))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                                  iterations=opened_iteration)
        #cv2.imshow('opened / iteration : {0}'.format(opened_iteration), opened)
        plt.imshow(opened, cmap='gray')
        plt.show()

    if key == ord('c'):
        closed_iteration = int(input('closed_iteration: '))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                                  iterations=closed_iteration)
        #cv2.imshow('closed / iteration : {0}'.format(closed_iteration), closed)
        plt.imshow(closed, cmap='gray')
        plt.show()

    if key == 27:
        print("ESC Pressed")
        break

cv2.destroyAllWindows()






