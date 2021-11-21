import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--path',default='./data/KakaoTalk_Photo_2021-09-08-20-29-00.png')
params = parser.parse_args()

img = cv2.imread(params.path)

assert img is not None

print('read {}'.format(params.path))
print('shape:', img.shape)
print('dtype:', img.dtype)

img = cv2.imread(params.path, cv2.IMREAD_GRAYSCALE)
assert img is not None

print('read {} as grayscale'.format(params.path))
print('shape:', img.shape)
print('dtype:', img.dtype)