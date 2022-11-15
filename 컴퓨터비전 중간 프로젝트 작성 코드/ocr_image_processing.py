import cv2
import numpy as np
import matplotlib.pyplot as plt

file_path  = './OCR데이터셋/dataset/classification/error_img/error_19_6_5.png'
img = cv2.imread(file_path, 0)
red_img = cv2.imread(file_path)

thr, mask = cv2.threshold(img, 200, 1, cv2.THRESH_BINARY)

plt.figure(figsize=(10,4))
plt.subplot(131)
plt.axis('off')
plt.title("original")
plt.imshow(red_img[:,:,[2,1,0]])
plt.subplot(132)
plt.axis('off')
plt.title("binary")
plt.imshow(img, cmap='gray')
plt.subplot(133)
plt.axis('off')
plt.title("binary threshold")
plt.imshow(mask, cmap='gray')
plt.tight_layout()
plt.show()