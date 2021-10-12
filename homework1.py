import cv2
import matplotlib.pyplot as plt
import numpy as np

# 1. 히스토그램 평탄화
img = cv2.imread('data/image_House256rgb.png')
image_to_show = np.copy(img)

cv2.namedWindow('house')
finish = False
while not finish:
    cv2.imshow("house", img)
    key = cv2.waitKey(1)

    # b 입력 시
    if key == ord('b'):
        image_to_show = cv2.equalizeHist(img[:, :, 0])
        cv2.imshow('equalized Blue channel', image_to_show)
        hist, bins = np.histogram(image_to_show, 256, [0, 255])
        plt.fill(hist)
        plt.xlabel('pixel value')
        plt.show()
    # g 입력 시
    if key == ord('g'):
        image_to_show = cv2.equalizeHist(img[:, :, 1])
        cv2.imshow('equalized Green Channel', image_to_show)
        hist, bins = np.histogram(image_to_show, 256, [0, 255])
        plt.fill(hist)
        plt.xlabel('pixel value')
        plt.show()
    # r 입력 시
    if key == ord('r'):
        image_to_show = cv2.equalizeHist(img[:, :, 2])
        cv2.imshow('equalized Red Channel', image_to_show)
        hist, bins = np.histogram(image_to_show, 256, [0, 255])
        plt.fill(hist)
        plt.xlabel('pixel value')
        plt.show()

    if key == 27:
        print("ESC Pressed")
        break

cv2.destroyAllWindows()