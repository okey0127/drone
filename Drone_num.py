import cv2
import numpy as np
import os
from flask import Flask, render_template, make_response
import matplotlib.pyplot as plt
import math

'''이와 같은 표시는 추후 실험으로 확정할 값들'''

# 카메라를 이용하여 실시간으로 이미지 파일 받아오기
cap = cv2.VideoCapture(0)

n = 0
gray_resize = np.zeros((50, 70))

# 현재 폴더 위치를 얻는 함수
now_dir = os.path.dirname(os.path.abspath(__file__))


# functions
def load_train_data(file_name):
    with np.load(file_name) as data:
        train = data['train']
        train_labels = data['train_labels']
    return train, train_labels

def resize120(image):
    global gray_resize
    gray_resize = cv2.resize(image, (50, 70))
    ret, gray_resize = cv2.threshold(gray_resize, 127, 255, cv2.THRESH_BINARY)
    # 최종적으로는 (1 x 3500) 크기로 반환합니다.
    return gray_resize.reshape(-1, 3500).astype(np.float32)

# KNN알고리즘으로 숫자 판별
def check(test, train, train_labels):
    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
    # 가장 가까운 5개의 글자를 찾아, 어떤 숫자에 해당하는지 찾는다.
    ret, result, neighbours, dist = knn.findNearest(test, k=5)
    return result


# 붉은 부분만 검출하기 위한 초기값들
'''실험을 통해 값 확정할 필요 있음'''
hsv = 0 # 기준 색의 위치
color_range = 7  # 빨간색으로 인식할 범위
threshold_S = 100  # 채도 하한값 (max 255)
threshold_V = 30  # 명도 하한값 (max 100)

cv2.namedWindow('img_mask')

print(now_dir)
while True:
    ret, frame = cap.read()
    if not ret:
        print("can't open camera")
        break
    # 카메라 보정?(수동으로 할지 직접할지 결정, 수동이면 camera_set 참고)

    lower_red1 = np.array([hsv - color_range + 180, threshold_S, threshold_V])
    upper_red1 = np.array([180, 255, 255])
    lower_red2 = np.array([0, threshold_S, threshold_V])
    upper_red2 = np.array([hsv, 255, 255])
    lower_red3 = np.array([hsv, threshold_S, threshold_V])
    upper_red3 = np.array([hsv + color_range, 255, 255])

    height, width, channel = frame.shape

    # 원본 영상을 HSV 영상으로 변환
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 범위 값으로 HSV 이미지에서 마스크를 생성합니다.
    img_mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
    img_mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
    img_mask3 = cv2.inRange(img_hsv, lower_red3, upper_red3)
    img_mask = img_mask1 | img_mask2 | img_mask3

    # 마스크 이미지로 원본 이미지에서 범위값에 해당되는 영상 부분을 획득합니다.
    img_result = cv2.bitwise_and(frame, frame, mask=img_mask)
    img_result = cv2.bitwise_not(img_result)  # 색반전
    img_gray = cv2.cvtColor(img_result, cv2.COLOR_BGR2GRAY)

    # 블러 처리를 통한 노이즈 제거
    img_blurred = cv2.GaussianBlur(img_gray, ksize=(15, 15), sigmaX=0)

    # 마스킹 영역(반전됨)의 구멍 제거(opening)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(img_blurred, cv2.MORPH_OPEN, kernel, 2)

    # 마스킹 영역(반전됨)의 팽창(erosion)
    kernal = np.ones((5, 3), np.uint8)
    erosion = cv2.erode(opening, kernal, iterations=2)

    # 팽창된 마스킹 영역 축소(dilation)
    kernal = np.ones((5, 3), np.uint8)
    dilation = cv2.dilate(erosion, kernal, iterations=2)

    # Thresholding
    ret, thresh = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY_INV)
    # thresh = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 9)

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    # temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    contours_dict = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # frame= cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 3)
        contours_dict.append({'contour': contour, 'x': x, 'y': y, 'w': w, 'h': h, 'cx': x + (w / 2), 'cy': y + (h / 2)})

    '''계속 실험해서 아래 수치 확립시킬 필요 있음'''
    Min_area = 300
    Min_width, Min_height = 10, 40
    min_ratio, max_ratio = 0.1, 1.5

    # 여기서 걸러진 것을 모두 '숫자' 라 가정
    possible_contours = []
    cnt = 0
    for d in contours_dict:
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']

        if area > Min_area and d['w'] > Min_width and d['h'] > Min_height and min_ratio < ratio < max_ratio:
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)

    for contour in possible_contours:
        x, y, w, h = cv2.boundingRect(contour['contour'])
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

    key_input = cv2.waitKey(300) # waitKey 함수로 프레임 조절 가능, 영상 송출 및 연산속도 느릴 시 숫자 키울것
    if key_input == ord('q'):
        break
    elif key_input == ord('a'):
        try:
            result = []
            for contour in possible_contours:
                # 이미지 크롭
                num_img = thresh[contour['y']:contour['y'] + contour['h'], contour['x']:contour['x'] + contour['w']]
                clearance_x = int(contour['h'] * 0.3)
                clearance_y = int(contour['h'] * 0.3)

                # 이미지에 여백을 준다
                num_img = cv2.copyMakeBorder(num_img, top=clearance_y, bottom=clearance_y, left=clearance_x,
                                             right=clearance_x, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

                # Adaptive Thresholding // 한번 더 이 작업을 수행하여 숫자의 형태를 분명하게 해준다.
                num_img_blurred = cv2.GaussianBlur(num_img, ksize=(5, 5), sigmaX=0)  # 노이즈 블러
                ret, num_thresh = cv2.threshold(num_img_blurred, 127, 255, cv2.THRESH_BINARY)
                # num_thresh = cv2.adaptiveThreshold(num_img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)

                plt.imshow(cv2.cvtColor(num_img, cv2.COLOR_BGR2RGB))
                plt.show()

                # KNN 머신러닝데이터로 대조하여 결과 출력
                FILE_NAME = now_dir + '/trained.npz'
                train, train_labels = load_train_data(FILE_NAME)

                # KNN
                test = resize120(num_thresh)
                dist_num = int(check(test, train, train_labels))
                result.append(int(dist_num))

            print(result)
        except:
            print('no number')

cap.release()
cv2.destroyAllWindows()