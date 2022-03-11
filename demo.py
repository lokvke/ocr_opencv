import cv2
import numpy as np
from api import text_detection, text_recognition


# 文本检测测试
def detection_test(file):
    image = cv2.imread(file)
    det_outputs = text_detection(image)
    print(det_outputs)
    show_det(det_outputs, image)


# 文本识别测试
def recognition_test(file):
    image = cv2.imread(file)
    rec_outputs = text_recognition(image)
    print(rec_outputs)


# 显示文本检测结果
def show_det(det_outputs, img_bgr):
    pts_list = []
    for res in det_outputs[0]:
        pts = np.expand_dims(res, axis=1)
        pts_list.append(pts)
        img_bgr = cv2.polylines(img_bgr, pts_list, True, (255, 0, 0), 2)
    cv2.imshow('windows', img_bgr)
    cv2.waitKey()


if __name__ == '__main__':
    file = 'data/jojo.jpg'
    detection_test(file)

    file2 = 'data/welcome.png'
    recognition_test(file2)
