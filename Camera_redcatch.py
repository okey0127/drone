import cv2
import numpy as np

# 글씨가 붉은 색이라 가정, 붉은 부분만 추려냄

hsv = 0
color_range = 10 # 빨간색으로 인식할 범위
threshold_S = 30 # 채도 하한값
threshold_V = 30 # 명도 하한값


lower_red1 = np.array([hsv - color_range + 180, threshold_S, threshold_V])
upper_red1 = np.array([180, 255, 255])
lower_red2 = np.array([0, threshold_S, threshold_V])
upper_red2 = np.array([hsv, 255, 255])
lower_red3 = np.array([hsv, threshold_S, threshold_V])
upper_red3 = np.array([hsv + color_range, 255, 255])


cap = cv2.VideoCapture(0)

while(True):
    ret, img_color = cap.read()
    height, width = img_color.shape[:2]
    #img_color = cv2.resize(img_color, (width, height), interpolation=cv2.INTER_AREA)

    # 원본 영상을 HSV 영상으로 변환합니다.
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

    # 범위 값으로 HSV 이미지에서 마스크를 생성합니다.
    img_mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
    img_mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
    img_mask3 = cv2.inRange(img_hsv, lower_red3, upper_red3)
    img_mask = img_mask1 | img_mask2 | img_mask3


    # 마스크 이미지로 원본 이미지에서 범위값에 해당되는 영상 부분을 획득합니다.
    img_result = cv2.bitwise_and(img_color, img_color, mask=img_mask)


    cv2.imshow('img_color', img_color)
    cv2.imshow('img_mask', img_mask)
    cv2.imshow('img_result', img_result)


    # ESC 키누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break


cv2.destroyAllWindows()