import cv2
import numpy as np
import matplotlib.pyplot as plt

def cal_distance(c_row, c_col, r, c):
    s = (c_row - r) ** 2 + (c_col - c) ** 2
    return s ** (1 / 2)


def filter_radius(fshift, rad, low=True):
    rows = fshift.shape[0]
    cols = fshift.shape[1]
    c_row, c_col = int(rows / 2), int(cols / 2)  # center

    filter_fshift = fshift.copy()

    for r in range(rows):
        for c in range(cols):
            if low:  # low-pass filter
                if cal_distance(c_row, c_col, r, c) > rad:
                    filter_fshift[r, c] = 0
            else:  # high-pass filter
                if cal_distance(c_row, c_col, r, c) < rad:
                    filter_fshift[r, c] = 0

    return filter_fshift


img = cv2.imread('data/Lena.png').astype(np.float32) / 255
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
fft = np.fft.fft2(gray)
fft_shift = np.fft.fftshift(fft, axes=[0,1])

# low-pass filter
low_fshift = filter_radius(fft_shift, rad=50, low=True)

# low-pass filter inverse fft
low_ishift = np.fft.ifftshift(low_fshift)
low_img = np.fft.ifft2(low_ishift)
low_img = np.abs(low_img)

plt.imshow(low_img, cmap='gray')
plt.show()

# high-pass filter
high_fshift = filter_radius(fft_shift, rad=50, low=False)

# high-pass filter inverse fft
high_ishift = np.fft.ifftshift(high_fshift)
high_img = np.fft.ifft2(high_ishift)
high_img = np.abs(high_img)

plt.imshow(high_img, cmap='gray')
plt.show()