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

image_to_show = np.copy(img)
LPF_mode = True
radius = 50

cv2.namedWindow('lena')
finish = False

while not finish:
    cv2.imshow("lena", image_to_show)
    key = cv2.waitKey(1)

    if key == ord('h'):
        LPF_mode = False
        fshift = filter_radius(fft_shift, rad=radius, low=LPF_mode)
        # inverse fft
        ifftshift = np.fft.ifftshift(fshift)
        transformed_img = np.fft.ifft2(ifftshift)
        transformed_img = np.abs(transformed_img)
        plt.imshow(transformed_img, cmap='gray')
        plt.show()

    if key == ord('l'):
        LPF_mode = True
        fshift = filter_radius(fft_shift, rad=radius, low=LPF_mode)
        # inverse fft
        ifftshift = np.fft.ifftshift(fshift)
        transformed_img = np.fft.ifft2(ifftshift)
        transformed_img = np.abs(transformed_img)
        plt.imshow(transformed_img, cmap='gray')
        plt.show()

    if key == ord('r'):
        radius = int(input("반지름 : "))
        fshift = filter_radius(fft_shift, rad=radius, low=LPF_mode)
        # inverse fft
        ifftshift = np.fft.ifftshift(fshift)
        transformed_img = np.fft.ifft2(ifftshift)
        transformed_img = np.abs(transformed_img)
        plt.imshow(transformed_img, cmap='gray')
        plt.show()

    if key == 27:
        print("ESC Pressed")
        break


cv2.destroyAllWindows()




