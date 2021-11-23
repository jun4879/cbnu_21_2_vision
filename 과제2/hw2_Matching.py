# 2. Matching
# stitching.zip에서 각 영상셋(boat, budapest, newspaper, s1~s2)에서 두 장을 선택하고
# 각 영상에서 각각 SIFT, SURF, ORB를 추출한 후에 매칭 및 RANSAC을 통해서
# 두 장의 영상간의 homography를 계산하고, 이를 통해 한 장의 영상을 다른 한 장의 영상으로 warping 하는 코드를 작성하시오.

import cv2
import numpy as np
import matplotlib.pyplot as plt

# image load
boat1_file_path = './data/boat1.jpg'
boat3_file_path = './data/boat3.jpg'
budapest1_file_path = './data/budapest1.jpg'
budapest5_file_path = './data/budapest5.jpg'
newspaper2_file_path = './data/newspaper2.jpg'
newspaper3_file_path = './data/newspaper3.jpg'
s1_file_path = './data/s1.jpg'
s2_file_path = './data/s2.jpg'

boat1_img = cv2.imread(boat1_file_path, cv2.IMREAD_GRAYSCALE)
boat3_img = cv2.imread(boat3_file_path, cv2.IMREAD_GRAYSCALE)
budapest1_img = cv2.imread(budapest1_file_path, cv2.IMREAD_GRAYSCALE)
budapest5_img = cv2.imread(budapest5_file_path, cv2.IMREAD_GRAYSCALE)
newspaper2_img = cv2.imread(newspaper2_file_path, cv2.IMREAD_GRAYSCALE)
newspaper3_img = cv2.imread(newspaper3_file_path, cv2.IMREAD_GRAYSCALE)
s1_img = cv2.imread(s1_file_path, cv2.IMREAD_GRAYSCALE)
s2_img = cv2.imread(s2_file_path, cv2.IMREAD_GRAYSCALE)

# sift / sufr / orb
sift = cv2.xfeatures2d.SIFT_create()
# surf = cv2.xfeatures2d.SURF_create()
orb = cv2.ORB_create()
# key point 추출
boat1_sift_kp, boat1_sift_des = sift.detectAndCompute(boat1_img, None)
boat3_sift_kp, boat3_sift_des = sift.detectAndCompute(boat3_img, None)
# boat1_surf_kp, boat1_surf_des = surf.detectAndCompute(boat1_img, None)
# boat3_surf_kp, boat3_surf_des = surf.detectAndCompute(boat3_img, None)
boat1_orb_kp, boat1_orb_des = orb.detectAndCompute(boat1_img, None)
boat3_orb_kp, boat3_orb_des = orb.detectAndCompute(boat3_img, None)

budapest1_sift_kp, budapest1_sift_des = sift.detectAndCompute(budapest1_img, None)
budapest5_sift_kp, budapest5_sift_des = sift.detectAndCompute(budapest5_img, None)
# budapest1_surf_kp, budapest1_surf_des = surf.detectAndCompute(budapest1_img, None)
# budapest5_surf_kp, budapest5_surf_des = surf.detectAndCompute(budapest5_img, None)
budapest1_orb_kp, budapest1_orb_des = orb.detectAndCompute(budapest1_img, None)
budapest5_orb_kp, budapest5_orb_des = orb.detectAndCompute(budapest5_img, None)

newspaper2_sift_kp, newspaper2_sift_des = sift.detectAndCompute(newspaper2_img, None)
newspaper3_sift_kp, newspaper3_sift_des = sift.detectAndCompute(newspaper3_img, None)
# newspaper2_surf_kp, newspaper2_surf_des = surf.detectAndCompute(newspaper2_img, None)
# newspaper3_surf_kp, newspaper3_surf_des = surf.detectAndCompute(newspaper3_img, None)
newspaper2_orb_kp, newspaper2_orb_des = orb.detectAndCompute(newspaper2_img, None)
newspaper3_orb_kp, newspaper3_orb_des = orb.detectAndCompute(newspaper3_img, None)

s1_sift_kp, s1_sift_des = sift.detectAndCompute(s1_img, None)
s2_sift_kp, s2_sift_des = sift.detectAndCompute(s2_img, None)
# s1_surf_kp, s1_surf_des = surf.detectAndCompute(s1_img, None)
# s2_surf_kp, s2_surf_des = surf.detectAndCompute(s2_img, None)
s1_orb_kp, s1_orb_des = orb.detectAndCompute(s1_img, None)
s2_orb_kp, s2_orb_des = orb.detectAndCompute(s2_img, None)

# BFMatcher
bf = cv2.BFMatcher_create()

boat_sift_matches = bf.match(boat1_sift_des, boat3_sift_des)
# boat_surf_matches = bf.match(boat1_surf_des, boat3_surf_des)
boat_orb_matches = bf.match(boat1_orb_des, boat3_orb_des)

budapest_sift_matches = bf.match(budapest1_sift_des, budapest5_sift_des)
# budapest_surf_matches = bf.match(budapest1_surf_des, budapest5_surf_des)
budapest_orb_matches = bf.match(budapest1_orb_des, budapest5_orb_des)

newspaper_sift_matches = bf.match(newspaper2_sift_des, newspaper3_sift_des)
# newspaper_surf_matches = bf.match(newspaper2_surf_des, newspaper3_surf_des)
newspaper_orb_matches = bf.match(newspaper2_orb_des, newspaper3_orb_des)

s_sift_matches = bf.match(s1_sift_des, s2_sift_des)
# s_surf_matches = bf.match(s1_surf_des, s2_surf_des)
s_orb_matches = bf.match(s1_orb_des, s2_orb_des)

# findHomography
boat_sift_pts0 = np.float32([boat1_sift_kp[m.queryIdx].pt for m in boat_sift_matches]).reshape(-1, 2)
boat_sift_pts1 = np.float32([boat3_sift_kp[m.queryIdx].pt for m in boat_sift_matches]).reshape(-1, 2)
# boat_surf_pts0 = np.float32([boat1_surf_kp[m.queryIdx].pt for m in boat_sift_matches]).reshape(-1, 2)
# boat_surf_pts1 = np.float32([boat3_surf_kp[m.queryIdx].pt for m in boat_sift_matches]).reshape(-1, 2)
boat_orb_pts0 = np.float32([boat1_orb_kp[m.queryIdx].pt for m in boat_sift_matches]).reshape(-1, 2)
boat_orb_pts1 = np.float32([boat3_orb_kp[m.queryIdx].pt for m in boat_sift_matches]).reshape(-1, 2)

budapest_sift_pts0 = np.float32([budapest1_sift_kp[m.queryIdx].pt for m in boat_sift_matches]).reshape(-1, 2)
budapest_sift_pts1 = np.float32([budapest5_sift_kp[m.queryIdx].pt for m in boat_sift_matches]).reshape(-1, 2)
# budapest_surf_pts0 = np.float32([budapest1_surf_kp[m.queryIdx].pt for m in boat_sift_matches]).reshape(-1, 2)
# budapest_surf_pts1 = np.float32([budapest5_surf_kp[m.queryIdx].pt for m in boat_sift_matches]).reshape(-1, 2)
budapest_orb_pts0 = np.float32([budapest1_orb_kp[m.queryIdx].pt for m in boat_sift_matches]).reshape(-1, 2)
budapest_orb_pts1 = np.float32([budapest5_orb_kp[m.queryIdx].pt for m in boat_sift_matches]).reshape(-1, 2)

newspaper_sift_pts0 = np.float32([newspaper2_sift_kp[m.queryIdx].pt for m in boat_sift_matches]).reshape(-1, 2)
newspaper_sift_pts1 = np.float32([newspaper3_sift_kp[m.queryIdx].pt for m in boat_sift_matches]).reshape(-1, 2)
# newspaper_surf_pts0 = np.float32([newspaper2_surf_kp[m.queryIdx].pt for m in boat_sift_matches]).reshape(-1, 2)
# newspaper_surf_pts1 = np.float32([newspaper3_surf_kp[m.queryIdx].pt for m in boat_sift_matches]).reshape(-1, 2)
newspaper_orb_pts0 = np.float32([newspaper2_orb_kp[m.queryIdx].pt for m in boat_sift_matches]).reshape(-1, 2)
newspaper_orb_pts1 = np.float32([newspaper3_orb_kp[m.queryIdx].pt for m in boat_sift_matches]).reshape(-1, 2)

s_sift_pts0 = np.float32([s1_sift_kp[m.queryIdx].pt for m in boat_sift_matches]).reshape(-1, 2)
s_sift_pts1 = np.float32([s2_sift_kp[m.queryIdx].pt for m in boat_sift_matches]).reshape(-1, 2)
# s_surf_pts0 = np.float32([s1_surf_kp[m.queryIdx].pt for m in boat_sift_matches]).reshape(-1, 2)
# s_surf_pts1 = np.float32([s2_surf_kp[m.queryIdx].pt for m in boat_sift_matches]).reshape(-1, 2)
s_orb_pts0 = np.float32([s1_orb_kp[m.queryIdx].pt for m in boat_sift_matches]).reshape(-1, 2)
s_orb_pts1 = np.float32([s2_orb_kp[m.queryIdx].pt for m in boat_sift_matches]).reshape(-1, 2)

boat_sift_H, _ = cv2.findHomography(boat_sift_pts0, boat_sift_pts1, cv2.RANSAC, 5.0)
# boat_surf_H, _ = cv2.findHomography(boat_surf_pts0, boat_surf_pts1, cv2.RANSAC, 5.0)
boat_orb_H, _ = cv2.findHomography(boat_orb_pts0, boat_orb_pts1, cv2.RANSAC, 5.0)

budapest_sift_H, _ = cv2.findHomography(budapest_sift_pts0, budapest_sift_pts1, cv2.RANSAC, 5.0)
# budapest_surf_H, _ = cv2.findHomography(budapest_surf_pts0, budapest_surf_pts1, cv2.RANSAC, 5.0)
budapest_orb_H, _ = cv2.findHomography(budapest_orb_pts0, budapest_orb_pts1, cv2.RANSAC, 5.0)

newspaper_sift_H, _ = cv2.findHomography(newspaper_sift_pts0, newspaper_sift_pts1, cv2.RANSAC, 5.0)
# newspaper_surf_H, _ = cv2.findHomography(newspaper_surf_pts0, newspaper_surf_pts1, cv2.RANSAC, 5.0)
newspaper_orb_H, _ = cv2.findHomography(newspaper_orb_pts0, newspaper_orb_pts1, cv2.RANSAC, 5.0)

s_sift_H, _ = cv2.findHomography(s_sift_pts0, s_sift_pts1, cv2.RANSAC, 5.0)
# s_surf_H, _ = cv2.findHomography(s_surf_pts0, s_surf_pts1, cv2.RANSAC, 5.0)
s_orb_H, _ = cv2.findHomography(s_orb_pts0, s_orb_pts1, cv2.RANSAC, 5.0)

boat_sift_warp = cv2.warpPerspective(boat3_img, boat_sift_H)
# boat_surf_warp = cv2.warpPerspective(boat3_img, boat_surf_H)
boat_orb_warp = cv2.warpPerspective(boat3_img, boat_orb_H)

budapest_sift_warp = cv2.warpPerspective(budapest5_img, budapest_sift_H)
# budapest_surf_warp = cv2.warpPerspective(budapest5_img, budapest_surf_H)
budapest_orb_warp = cv2.warpPerspective(budapest5_img, budapest_orb_H)

newspaper_sift_warp = cv2.warpPerspective(newspaper3_img, newspaper_sift_H)
# newspaper_surf_warp = cv2.warpPerspective(newspaper3_img, newspaper_surf_H)
newspaper_orb_warp = cv2.warpPerspective(newspaper3_img, newspaper_orb_H)

s_sift_warp = cv2.warpPerspective(s2_img, s_sift_H)
# s_surf_warp = cv2.warpPerspective(s2_img, s_surf_H)
s_orb_warp = cv2.warpPerspective(s2_img, s_orb_H)

plt.figure()
plt.subplot(431)
plt.axis('off')
plt.title('boat sift')
plt.imshow(boat_sift_warp)
plt.subplot(432)
plt.axis('off')
plt.title('boat surf')
# plt.imshow(boat_surf_warp)
plt.subplot(433)
plt.axis('off')
plt.title('boat orb')
plt.imshow(boat_orb_warp)
plt.subplot(434)
plt.axis('off')
plt.title('budapest sift')
plt.imshow(budapest_sift_warp)
plt.subplot(435)
plt.axis('off')
plt.title('budapest surf')
# plt.imshow(budapest_surf_warp)
plt.subplot(436)
plt.axis('off')
plt.title('budapest orb')
plt.imshow(budapest_orb_warp)
plt.subplot(437)
plt.axis('off')
plt.title('newspaper sift')
plt.imshow(newspaper_sift_warp)
plt.subplot(438)
plt.axis('off')
plt.title('newspaper surf')
# plt.imshow(newspaper_surf_warp)
plt.subplot(439)
plt.axis('off')
plt.title('newspaper orb')
plt.imshow(newspaper_orb_warp)
plt.subplot(4310)
plt.axis('off')
plt.title('S sift')
plt.imshow(s_sift_warp)
plt.subplot(4311)
plt.axis('off')
plt.title('S surf')
# plt.imshow(s_surf_warp)
plt.subplot(4312)
plt.axis('off')
plt.title('S orb')
plt.imshow(s_orb_warp)
