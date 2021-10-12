import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('data/image_Peppers512rgb.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, -1, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# eroded_iteration = int(input('eroded_iteration: '))
eroded = cv2.morphologyEx(binary, cv2.MORPH_ERODE, (3,3), iterations=10)
dilated = cv2.morphologyEx(binary, cv2.MORPH_DILATE, (3,3), iterations=10)

opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),
                          iterations=5)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),
                          iterations=5)

plt.figure(figsize=(10,10))
plt.subplot(231)
plt.axis('off')
plt.title('binary')
plt.imshow(binary, cmap='gray')
plt.subplot(232)
plt.axis('off')
plt.title('erode 10 times')
plt.imshow(eroded, cmap='gray')
plt.subplot(233)
plt.axis('off')
plt.title('dilate 10 times')
plt.imshow(dilated, cmap='gray')
plt.subplot(234)
plt.axis('off')
plt.title('open 5 times')
plt.imshow(opened, cmap='gray')
plt.subplot(235)
plt.axis('off')
plt.title('closed 5 times')
plt.imshow(closed, cmap='gray')
plt.tight_layout()
plt.show()
