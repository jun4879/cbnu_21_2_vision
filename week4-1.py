import cv2
import matplotlib.pyplot as plt

img = cv2.imread('data/KakaoTalk_Photo_2021-09-08-20-29-00.png',0)

dx = cv2.Sobel(img,cv2.CV_32F,1,0)
dy = cv2.Sobel(img,cv2.CV_32F,0,1)

plt.figure(figsize=(8,3))
plt.subplot(131)
plt.axis('off')
plt.title("image")
plt.imshow(img, cmap='gray')
plt.subplot(132)
plt.axis('off')
plt.imshow(dx, cmap='gray')
plt.title(r'$\frac{dI}{dx}$')
plt.subplot(133)
plt.axis('off')
plt.imshow(dy, cmap='gray')
plt.title(r'$\frac{dI}{dy}$')
plt.tight_layout()
plt.show()
