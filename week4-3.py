import cv2
import numpy as np
import matplotlib.pyplot as plt

# gray 이미지 넣어야 8번 라인에서 오류 없음
img = cv2.imread('data/KakaoTalk_Photo_2021-09-08-20-29-00.png', 0).astype(np.float32) / 255

fft = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
fft_shift = np.fft.fftshift(fft,axes=[0,1])

sz = 25
mask = np.zeros(fft.shape, np.uint8)
mask[img.shape[0]//2-sz:img.shape[0]//2+sz,
     img.shape[1]//2-sz:img.shape[1]//2+sz, :] = 1
fft_shift *= mask
fft = np.fft.ifftshift(fft_shift, axes=[0,1])

filtered = cv2.idft(fft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
mask_new = np.dstack((mask, np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)))

plt.figure()
plt.subplot(131)
plt.axis('off')
plt.title("original")
plt.imshow(img, cmap='gray')
plt.subplot(132)
plt.axis('off')
plt.title("no high frequency")
plt.imshow(filtered, cmap='gray')
plt.subplot(133)
plt.axis('off')
plt.title("mask")
plt.imshow(mask_new*255, cmap='gray')
plt.tight_layout(True)
plt.show()