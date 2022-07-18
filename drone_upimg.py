import cv2
import numpy as np
import os
from flask import Flask, render_template, make_response
import matplotlib.pyplot as plt
import math
import pytesseract
from PIL import Image
####학습 데이터 수집을 위한 코드#### 
pytesseract.pytesseract.tesseract_cmd=R'C:\Program Files\Tesseract-OCR\tesseract.exe'

### 1. 캠을 통한 영상 송출 및 캡쳐 except web streaming
cap = cv2.VideoCapture(0)

n=0

#현재 폴더 위치를 얻는 함수
now_dir = os.path.dirname(os.path.abspath(__file__))

#functions
def load_train_data(file_name):
    with np.load(file_name) as data:
        train = data['train']
        train_labels = data['train_labels']
    return train, train_labels

def resize120(image):
    global n
    gray_resize = cv2.resize(image, (50, 70))
    save_file = '/number/up_image/{}number.jpg'.format(n)
    while True:
        if os.path.exists(now_dir+save_file):
            n += 1
            save_file = '/number/up_image/{}number.jpg'.format(n)
        else:
            break
        
    cv2.imwrite(now_dir+save_file, gray_resize)
    # 최종적으로는 (1 x 3500) 크기로 반환합니다.
    return gray_resize.reshape(-1, 3500).astype(np.float32)

def check(test, train, train_labels):
    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
    # 가장 가까운 5개의 글자를 찾아, 어떤 숫자에 해당하는지 찾는다.
    ret, result, neighbours, dist = knn.findNearest(test, k=5)
    return result

print(now_dir)
while True:
    ret, frame = cap.read()
    if not ret:
        print("can't open camera")
        break
    
    
    height, width, channel = frame.shape
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Adaptive Thresholding
    img_blurred = cv2.GaussianBlur(img_gray, ksize=(5,5), sigmaX=0) #노이즈 블러
    thresh = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9) 
    cv2.imshow('th', thresh)

    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
        
    contours_dict =[]
        
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        temp_result = cv2.rectangle(temp_result, (x, y), (x+w, y+h), (0,0,255), 3)
        contours_dict.append({'contour':contour, 'x':x, 'y':y, 'w':w, 'h':h, 'cx': x+(w/2), 'cy': y+(h/2)})
    
    Min_area = 500
    Min_width, Min_height = 10, 40
    min_ratio, max_ratio = 0.1, 1
    
    possible_contours = []
    cnt = 0
    for d in contours_dict:
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']
            
        if area > Min_area and d['w'] > Min_width and d['h'] > Min_height and min_ratio < ratio < max_ratio:
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)    
    
    #사각형 중심이 이미지의 중앙에 가장 가까운 것을 '숫자' 로 인식한다.
    diff = 1000
    for d in possible_contours:
        d_diff = math.sqrt((d['cx']-width/2)**2+(d['cy']-height/2)**2)
        if d_diff < diff:
            diff = d_diff
            n_contour = d
    try:   
        clearance_x = int(n_contour['h']*0.3)
        clearance_y = int(n_contour['h']*0.3)
        
        frame = cv2.rectangle(frame, (n_contour['x'], n_contour['y']), (n_contour['x']+n_contour['w'], n_contour['y']+n_contour['h']), (255, 0, 0), 3)
        cv2.imshow('test', frame)
    except:
        
        cv2.imshow('test', frame)
        
    key_input = cv2.waitKey(1)
    if key_input == ord('q'):
        break
    elif key_input == ord('a'):
        try:
            #이미지 크롭
            num_img = thresh[n_contour['y']:n_contour['y']+n_contour['h'],n_contour['x']:n_contour['x']+n_contour['w']]
            #이미지에 여백을 준다
            num_img= cv2.copyMakeBorder(num_img, top=clearance_y, bottom=clearance_y, left=clearance_x, right=clearance_x, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
            #Adaptive Thresholding // 한번 더 이 작업을 수행하여 숫자의 형태를 분명하게 해준다.
            num_img_blurred = cv2.GaussianBlur(num_img, ksize=(5,5), sigmaX=0) #노이즈 블러
            num_thresh = cv2.adaptiveThreshold(num_img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9) 
        
            cv2.imwrite(now_dir+'/number.jpg', num_thresh)
            #plt.imshow(cv2.cvtColor(num_img, cv2.COLOR_BGR2RGB))
            #plt.show()
            
            #KNN 머신러닝데이터로 대조하여 결과 출력
            FILE_NAME = now_dir + '/number/up_trained.npz'
            train, train_labels = load_train_data(FILE_NAME)
            #KNN
            test = resize120(num_img)
            result = check(test, train, train_labels)
            print(int(result))
            '''
            #tesseract
            image = Image.open(now_dir+'/number.jpg')
            dd = pytesseract.image_to_string(image, lang=None)
            print(dd)
            '''
        except:
            
            print('no number')

cap.release()
cv2.destroyAllWindows()