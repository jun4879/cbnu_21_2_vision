import cv2
import numpy as np
import matplotlib.pyplot as plt

gray = cv2.imread('./data/KakaoTalk_Photo_2021-09-08-20-29-00.png',0)
cv2.imshow('original image', gray)
cv2.waitKey()

hist, bins = np.histogram(gray, 256, [0,255])
plt.fill(hist)
plt.xlabel('pixel value')
plt.show()

gray_eq = cv2.equalizeHist(gray)
hist, bins = np.histogram(gray_eq, 256, [0,255])
plt.fill_between(range(256), hist, 0)
plt.xlabel('pixel value')
plt.show()

cv2.imshow('equlized gray',gray_eq)
cv2.waitKey()

color = cv2.imread('./data/KakaoTalk_Photo_2021-09-08-20-29-00.png')
hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

hsv[...,2] = cv2.equalizeHist(hsv[...,2])
color_eq = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imshow('original color', color)
cv2.imshow('equalized color', color_eq)
cv2.waitKey()
cv2.destroyAllWindows()


