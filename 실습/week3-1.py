import cv2
import argparse
import numpy as np
try:
    from cv2 import cv2
except ImportError:
    pass

parser = argparse.ArgumentParser()
parser.add_argument('--path',default='./data/KakaoTalk_Photo_2021-09-08-20-29-00.png',
                    help='Image path.')
parser.add_argument('--out_jpg',default='./data/lena_draw.jpg',
                    help='draw objects')
params = parser.parse_args()
img = cv2.imread(params.path)


assert img is not None

global image_to_show, s_x, s_y, e_x, e_y, mouse_pressed
global line_mode, arrow_mode, rec_mode

image_to_show = np.copy(img)
line_mode = True
arrow_mode = False
rec_mode = False

def mouse_callback(event, x, y, flags, params):
    global s_x, s_y, e_x, e_y, mouse_pressed
    global line_mode, arrow_mode, rec_mode

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        s_x, s_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False
        e_x, e_y = x, y
        if line_mode:
            cv2.line(image_to_show, (s_x,s_y), (e_x,e_y), (255,0,0), 3 )
        if arrow_mode:
            cv2.arrowedLine(image_to_show, (s_x,s_y), (e_x,e_y), (255,255,0), 3 )
        if rec_mode:
            cv2.rectangle(image_to_show, (s_x,s_y), (e_x,e_y), (255,0,255), 3)


cv2.namedWindow('lena')
cv2.setMouseCallback('lena',mouse_callback)
finish = False

while not finish:
    cv2.imshow("lena", image_to_show)
    key = cv2.waitKey(1)

    if key == ord('l'):
        line_mode = True
        arrow_mode = False
        rec_mode = False
    if key == ord('a'):
        line_mode = False
        arrow_mode = True
        rec_mode = False
    if key == ord('r'):
        line_mode = False
        arrow_mode = False
        rec_mode = True

    if key == ord('c'):
        image_to_show = np.copy(img)

    if key == ord('w'):
        print("img saved")
        cv2.imwrite(params.out_jpg, img)

    if key == 27:
        print("ESC Pressed")
        break

cv2.destroyAllWindows()