import cv2
import numpy as np
import matplotlib.pyplot as plt

file_path  = './OCR데이터셋/dataset/classification/error_img/error_17_0_8.png'
binary = cv2.imread(file_path, 0)
red_img = cv2.imread(file_path)

KSIZE = 11
ALPHA = 3
kernel = cv2.getGaussianKernel(KSIZE,0)
kernel = - ALPHA * kernel @ kernel.T
kernel[KSIZE//2, KSIZE//2] += 1+ALPHA

filtered = cv2.filter2D(binary, -1, kernel)
thr, mask = cv2.threshold(filtered, -1, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

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
plt.title("filtered")
plt.imshow(filtered, cmap='gray')
plt.subplot(144)
plt.axis('off')
plt.title("otsu")
plt.imshow(mask, cmap='gray')
plt.tight_layout()
plt.show()
