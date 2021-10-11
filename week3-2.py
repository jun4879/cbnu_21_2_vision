import cv2
import numpy as np

img = cv2.imread('data/KakaoTalk_Photo_2021-09-08-20-29-00.png').astype(np.float32) / 255
print('shape: ', img.shape)
print('dtype: ', img.dtype)
cv2.imshow('original image',img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("converted to gray")
print('shape: ', gray.shape)
print('dtype: ', gray.dtype)
cv2.imshow('gray image',gray)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
print("converted to hsv")
print('shape: ', hsv.shape)
print('dtype: ', hsv.dtype)
cv2.imshow('hsv image',hsv)

hsv[:,:,2] *= 2
from_hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
print("converted to bgr from hsv")
print('shape: ', from_hsv.shape)
print('dtype: ', from_hsv.dtype)
cv2.imshow('from_hsv', from_hsv)
cv2.waitKey()
cv2.destroyAllWindows()