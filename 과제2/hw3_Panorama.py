# 3. Panorama
# CreaterStitcher 함수를 이용하여 4개의 영상 셋에 대해서 파노라마 이미지를 만드는 방법을 구현하시오.

import cv2

boat_images = []
boat_images.append(cv2.imread('./data/boat1.jpg', cv2.IMREAD_COLOR))
boat_images.append(cv2.imread('./data/boat2.jpg', cv2.IMREAD_COLOR))
boat_images.append(cv2.imread('./data/boat3.jpg', cv2.IMREAD_COLOR))
boat_images.append(cv2.imread('./data/boat4.jpg', cv2.IMREAD_COLOR))
boat_images.append(cv2.imread('./data/boat5.jpg', cv2.IMREAD_COLOR))
boat_images.append(cv2.imread('./data/boat6.jpg', cv2.IMREAD_COLOR))

budapest_images = []
budapest_images.append(cv2.imread('./data/budapest1.jpg', cv2.IMREAD_COLOR))
budapest_images.append(cv2.imread('./data/budapest2.jpg', cv2.IMREAD_COLOR))
budapest_images.append(cv2.imread('./data/budapest3.jpg', cv2.IMREAD_COLOR))
budapest_images.append(cv2.imread('./data/budapest4.jpg', cv2.IMREAD_COLOR))
budapest_images.append(cv2.imread('./data/budapest5.jpg', cv2.IMREAD_COLOR))
budapest_images.append(cv2.imread('./data/budapest6.jpg', cv2.IMREAD_COLOR))

newspaper_images = []
newspaper_images.append(cv2.imread('./data/newspaper1.jpg', cv2.IMREAD_COLOR))
newspaper_images.append(cv2.imread('./data/newspaper2.jpg', cv2.IMREAD_COLOR))
newspaper_images.append(cv2.imread('./data/newspaper3.jpg', cv2.IMREAD_COLOR))
newspaper_images.append(cv2.imread('./data/newspaper4.jpg', cv2.IMREAD_COLOR))

s_images = []
s_images.append(cv2.imread('./data/s1.jpg', cv2.IMREAD_COLOR))
s_images.append(cv2.imread('./data/s2.jpg', cv2.IMREAD_COLOR))

stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)

boat_ret, boat_pano = stitcher.stitch(boat_images)
budapest_ret, budapest_pano = stitcher.stitch(budapest_images)
newspaper_ret, newspaper_pano = stitcher.stitch(newspaper_images)
s_ret, s_pano = stitcher.stitch(s_images)

boat_pano_res = cv2.resize(boat_pano, dsize=(0, 0), fx=0.1, fy=0.1)
cv2.imshow('boat panorama', boat_pano_res)
cv2.imshow('budapest panorama', budapest_pano)
cv2.imshow('newspaper panorama', newspaper_pano)
cv2.imshow('s panorama', s_pano)

cv2.waitKey()
cv2.destroyAllWindows()
