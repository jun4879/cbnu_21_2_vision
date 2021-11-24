# dog 영상을 이용해서 Good Feature to Tracking을 추출하고 Pyramid Lucas-Kanade 알고리즘을 적용해서 Optical Flow를 구하는 코드를 작성하시오.
# dog 영상을 이용해서 Farneback과 DualTVL1 Optical Flow 알고리즘을 구하는 코드를 작성하시오.

import cv2
import numpy as np

dog_A_img = cv2.imread('./data/dog_a.jpg')
dog_B_img = cv2.imread('./data/dog_b.jpg')

dog_A_gray = cv2.cvtColor(dog_A_img, cv2.COLOR_BGR2GRAY)
dog_B_gray = cv2.cvtColor(dog_B_img, cv2.COLOR_BGR2GRAY)

color = np.random.randint(0, 255, (200, 3))
lines = np.zeros_like(dog_A_gray)
termcriteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

pts = cv2.goodFeaturesToTrack(dog_A_gray, 200, 0.01, 10)
pts = pts.reshape(-1, 1, 2)
prevPt = pts
nextPt, status, errors = cv2.calcOpticalFlowPyrLK(dog_A_gray, dog_B_gray, prevPt, None, criteria=termcriteria)
prevMv = prevPt[status == 1]
nextMv = nextPt[status == 1]

for i, (p, n) in enumerate(zip(prevMv, nextMv)):
    px, py = p.ravel()
    nx, ny = n.ravel()
    cv2.line(lines, (px, py), (nx, ny), color[i].tolist(), 2)
    cv2.circle(dog_B_gray, (nx, ny), 2, color[i].tolist(), -1)

# Lucas-Kanade 영상 출력
dog_LK_img = cv2.add(dog_A_gray, lines)
cv2.imshow("OpticalFlow - Lucas-Kanade", dog_LK_img)
cv2.waitKey()
cv2.destroyAllWindows()


def display_flow(img, flow, stride=40):
    for index in np.ndindex(flow[::stride, ::stride].shape[:2]):
        pt1 = tuple(i * stride for i in index)
        delta = flow[pt1].astype(np.int32)[::-1]
        pt2 = tuple(pt1 + 10 * delta)
        if 2 <= cv2.norm(delta) <= 10:
            cv2.arrowedLine(img, pt1[::-1], pt2[::-1], (0, 0, 255), 5, cv2.LINE_AA, 0, 0.4)
    norm_opt_flow = np.linalg.norm(flow, axis=2)
    norm_opt_flow = cv2.normalize(norm_opt_flow, None, 0, 1, cv2.NORM_MINMAX)

    cv2.imshow('OpticalFlow', norm_opt_flow)
    cv2.waitKey()
    cv2.destroyAllWindows()

# Farneback 영상 출력
farnebackFlow = cv2.calcOpticalFlowFarneback(dog_A_gray, dog_B_gray, None, 0.5, 3, 15, 3, 5, 1.1,
                                             cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
display_flow(dog_A_gray, farnebackFlow)
