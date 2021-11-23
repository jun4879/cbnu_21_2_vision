# 1. Feature Detection
# stitching.zip에서 4장의 영상(boat1, budapest1, newpaper1, s1)을 선택한 후에
# Canny Edge와 Harris Corner를 검출해서 결과를 출력하는 코드를 작성하시오.
import cv2
import matplotlib.pyplot as plt

# image load
boat_file_path = './data/boat1.jpg'
budapest_file_path = './data/budapest1.jpg'
newspaper_file_path = './data/newspaper1.jpg'
s_file_path = './data/s1.jpg'

boat_img = cv2.imread(boat_file_path, cv2.IMREAD_GRAYSCALE)
budapest_img = cv2.imread(budapest_file_path, cv2.IMREAD_GRAYSCALE)
newspaper_img = cv2.imread(newspaper_file_path, cv2.IMREAD_GRAYSCALE)
s_img = cv2.imread(s_file_path, cv2.IMREAD_GRAYSCALE)

# Canny Edge - 수치 조절하기
boat_canny = cv2.Canny(boat_img, 50, 200)
budapest_canny = cv2.Canny(budapest_img, 50, 200)
newspaper_canny = cv2.Canny(newspaper_img, 50, 200)
s_canny = cv2.Canny(s_img, 50, 200)

plt.figure()
plt.subplot(141)
plt.axis('off')
plt.title("boat canny edge")
plt.imshow(boat_canny, cmap='gray')
plt.subplot(142)
plt.axis('off')
plt.title("budapest canny edge")
plt.imshow(budapest_canny, cmap='gray')
plt.subplot(143)
plt.axis('off')
plt.title("newspaper canny edge")
plt.imshow(newspaper_canny, cmap='gray')
plt.subplot(144)
plt.axis('off')
plt.title("s canny edge")
plt.imshow(s_canny, cmap='gray')
plt.tight_layout()
plt.show()

# Harris Corner - 수치 조절하기
boat_harris = cv2.cornerHarris(boat_img, 2, 3, 0.04)
budapest_harris = cv2.cornerHarris(budapest_img, 2, 3, 0.04)
newspaper_harris = cv2.cornerHarris(newspaper_img, 2, 3, 0.04)
s_harris = cv2.cornerHarris(s_img, 2, 3, 0.04)

plt.figure()
plt.subplot(141)
plt.axis('off')
plt.title("boat harris corner")
plt.imshow(boat_harris, cmap='gray')
plt.subplot(142)
plt.axis('off')
plt.title("budapest harris corner")
plt.imshow(budapest_harris, cmap='gray')
plt.subplot(143)
plt.axis('off')
plt.title("newspaper harris corner")
plt.imshow(newspaper_harris, cmap='gray')
plt.subplot(144)
plt.axis('off')
plt.title("s harris corner")
plt.imshow(s_harris, cmap='gray')
plt.tight_layout()
plt.show()
