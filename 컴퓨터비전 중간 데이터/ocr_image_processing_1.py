import cv2
import numpy as np
import matplotlib.pyplot as plt

file_path  = './OCR데이터셋/dataset/classification/error_img/error_17_0_8.png'
binary = cv2.imread(file_path, 0)
red_img = cv2.imread(file_path)

thr, mask = cv2.threshold(binary, -1, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
eroded = cv2.morphologyEx(mask, cv2.MORPH_ERODE, (5, 5), iterations=5)

plt.figure(figsize=(10,4))
plt.subplot(141)
plt.axis('off')
plt.title("original")
plt.imshow(red_img[:,:,[2,1,0]])
plt.subplot(142)
plt.axis('off')
plt.title("binary")
plt.imshow(binary, cmap='gray')
plt.subplot(143)
plt.axis('off')
plt.title("otsu")
plt.imshow(mask, cmap='gray')
plt.subplot(144)
plt.axis('off')
plt.title("otsu eroded")
plt.imshow(eroded, cmap='gray')
plt.tight_layout()
plt.show()
