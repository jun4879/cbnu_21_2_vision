import cv2
import matplotlib.pyplot as plt
import numpy as np

# 2. 공간 도메인 필터
img = cv2.imread('data/Lena.png').astype(np.float32) / 255
noised = (img + 0.2 * np.random.rand(*img.shape).astype(np.float32))
noised = noised.clip(0,1)

# diameter : 필터링에 사용될 이웃 픽셀의 거리
d = 0
# SigmaColor : 색 공간에서의 필터의 표준 편차, 클수록 가우시안 필터
s = 0
# SigmaSpace : 좌표 공간에서의 필터의 표준편차
s = 0

image_to_show = np.copy(noised)
cv2.namedWindow('bilateralFilter')
finish = False
while not finish:
    cv2.imshow('bilateralFilter', noised)
    key = cv2.waitKey(1)

    if key == ord('q'):
        d = int(input("d : "))
        s = int(input("s : "))
        c = int(input("c : "))
        image_to_show = cv2.bilateralFilter(noised,d,s,c)
        cv2.imshow('Filtered Image', image_to_show)

    if key == 27:
        print("ESC Pressed")
        break

cv2.destroyAllWindows()