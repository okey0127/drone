import cv2
import numpy as np
import os
from flask import Flask, render_template, make_response, redirect, Response, request, jsonify
import matplotlib.pyplot as plt
import math
import threading 
import json
import requests
import time
import socket
### ver.2 red_detect version edit: 22.08.21 ###
## 객체 검출 ON/OFF 및 송출 영상 GRAY ##

img_w = 640
img_h = 480

global detect_result
detect_result = '-'

global object_detect
object_detect = "OFF"

global number_detect
number_detect = 'N'

global num_ex
num_ex = 'N'

global threshold_S

global threshold_V

global threshold
threshold = 235

# Image frame sent to the Flask object
global video_frame
video_frame = None

# Use locks for thread-safe viewing of frames in multiple browsers
global thread_lock 
thread_lock = threading.Lock()

#인터넷 접속 될때 까지 무한루프를 돌린다.
i_flag = 'Y'
while True:
    try:
        #ip 주소 수집(외부)
        URL = 'https://icanhazip.com'
        respons = requests.get(URL)
        ex_ip = respons.text.strip()
        ex_ip_video = 'http://'+ex_ip+':8080/video'
        break
    except:
        # 경고문은 추후 LCD로 출력되도록
        if i_flag == 'Y':
            print('No internet')
            i_flag = 'N'

# 리눅스에서 os.popen('hostname -I').read().strip() (내부)
in_ip = socket.gethostbyname(socket.gethostname())
in_ip_video = 'http://'+in_ip+':8080/video'

in_ipaddr = {'ip':in_ip, 'video':in_ip_video}
ex_ipaddr = {'ip':ex_ip, 'video':ex_ip_video}

print(in_ip, ex_ip)
#현재 폴더 위치 획득
now_dir = os.path.dirname(os.path.abspath(__file__))

'''수정 요함'''
# 붉은 부분만 검출하기 위한 초기값들
hsv = 0
color_range = 9 # 빨간색으로 인식할 범위
threshold_S = 80 # 채도 하한값
threshold_V = 30 # 명도 하한값


lower_red1 = np.array([hsv - color_range + 180, threshold_S, threshold_V])
upper_red1 = np.array([180, 255, 255])
lower_red2 = np.array([0, threshold_S, threshold_V])
upper_red2 = np.array([hsv, 255, 255])
lower_red3 = np.array([hsv, threshold_S, threshold_V])
upper_red3 = np.array([hsv + color_range, 255, 255])

#functions
def load_train_data(file_name):
    with np.load(file_name) as data:
        train = data['train']
        train_labels = data['train_labels']
    return train, train_labels

n = 1

def resize120(image):
    global n
    gray_resize = cv2.resize(image, (50, 70))
    save_file = '/number/image/{}number.jpg'.format(n)
    while True:
        if os.path.exists(now_dir+save_file):
            n += 1
            save_file = '/number/image/{}number.jpg'.format(n)
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

# Flask server 선언
app = Flask(__name__)

# main loop code -- 영상 스트리밍, 이미지 처리, 이미지 수집 --
def captureFrames():
    global video_frame, thread_lock, number_detect, detect_result, num_ex
    #아래 코드는 윈도우에서 쓸때로 리눅스에선 cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_h)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            #오류 페이지 생성
            return '<h1>Error:</h1> <p>Camera is not opened...</p>'
        
        height, width, channel = frame.shape
        
        if object_detect == 'ON':
            # 원본 영상을 HSV 영상으로 변환
            img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # 범위 값으로 HSV 이미지에서 마스크를 생성합니다.
            img_mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
            img_mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
            img_mask3 = cv2.inRange(img_hsv, lower_red3, upper_red3)
            img_mask = img_mask1 | img_mask2 | img_mask3
    
            # 마스크 이미지로 원본 이미지에서 범위값에 해당되는 영상 부분을 획득합니다.
            img_result = cv2.bitwise_and(frame, frame, mask=img_mask)
            img_result = cv2.bitwise_not(img_result) #색반전
            img_gray = cv2.cvtColor(img_result, cv2.COLOR_BGR2GRAY)

            # 블러 처리를 통한 노이즈 제거
            img_blurred = cv2.GaussianBlur(img_gray, ksize=(15,15), sigmaX=0) 
        
            # 체크박스 체크하면 구멍 제거 코드 실행  연산이 많아짐을 우려
            if num_ex == 'Y':
                # 마스킹 영역(반전됨)의 구멍 제거(opening)
                kernel = np.ones((3, 3), np.uint8)
                img_blurred = cv2.morphologyEx(img_blurred, cv2.MORPH_OPEN, kernel, 2)

                # 마스킹 영역(반전됨)의 팽창(erosion)
                kernal = np.ones((5, 3), np.uint8)
                img_blurred = cv2.erode(img_blurred, kernal, iterations=2)
    
                # 팽창된 마스킹 영역 축소(dilation)
                kernal = np.ones((5, 3), np.uint8)
                img_blurred = cv2.dilate(img_blurred, kernal, iterations=2)

            # Thresholding
            ret, thresh = cv2.threshold(img_blurred, threshold, 255, cv2.THRESH_BINARY_INV)
            #thresh = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9) 

            contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            #temp_result = np.zeros((height, width, channel), dtype=np.uint8)
        
            contours_dict =[]
        
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                #temp_result = cv2.rectangle(temp_result, (x, y), (x+w, y+h), (0,0,255), 3)
                contours_dict.append({'contour':contour, 'x':x, 'y':y, 'w':w, 'h':h, 'cx': x+(w/2), 'cy': y+(h/2)})

            '''수정 요함'''
            Min_area = 300
            Min_width, Min_height = 10, 40
            min_ratio, max_ratio = 0.1, 1.5
    
            possible_contours = []
            for d in contours_dict:
                area = d['w'] * d['h']
                ratio = d['w'] / d['h']
            
                if area > Min_area and d['w'] > Min_width and d['h'] > Min_height and min_ratio < ratio < max_ratio:
                    possible_contours.append(d)    
            
        try:
            for contour in possible_contours:
                x, y, w, h = cv2.boundingRect(contour['contour'])
                # 송출 영상
                frame= cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,0), 5)
                frame= cv2.rectangle(frame, (x, y), (x+w, y+h), (255,255,255), 3)
            with thread_lock:
                video = frame.copy()
                cv2.line(video, (310,240),(330,240), (0,0,0), 5)
                cv2.line(video, (320,235),(320,285), (0,0,0), 5)
                cv2.line(video, (310,240),(330,240), (255,255,255), 3)
                cv2.line(video, (320,235),(320,285), (255,255,255), 3)
                video = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
                video_frame = cv2.resize(video, None, fx = 0.5, fy=0.5, interpolation = cv2.INTER_AREA)
            if object_detect == 'OFF':
                possible_contours = []
        except:
            with thread_lock:
                video = frame.copy()
                cv2.line(video, (310,240),(330,240), (0,0,0), 5)
                cv2.line(video, (320,235),(320,285), (0,0,0), 5)
                cv2.line(video, (310,240),(330,240), (255,255,255), 3)
                cv2.line(video, (320,235),(320,285), (255,255,255), 3)
                video = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
                video_frame = cv2.resize(video, None, fx = 0.5, fy=0.5, interpolation = cv2.INTER_AREA)
                
        # 코드에 딜레이를 줘서 연산량을 줄인다. 이것으로 라즈베리 파이의 속도 개선이 되는지 확인
        #time.sleep(0.2)
        
        # 숫자 판별
        if number_detect == 'Y':
            try:
                detect_result = []
                if len(possible_contours) == 0:
                    raise Exception('No number')
                for contour in possible_contours:
                    #이미지 크롭
                    num_img = thresh[contour['y']:contour['y']+contour['h'],contour['x']:contour['x']+contour['w']]
                    clearance_x = int(contour['h']*0.3)
                    clearance_y = int(contour['h']*0.3)
                
                    #이미지에 여백을 준다
                    num_img= cv2.copyMakeBorder(num_img, top=clearance_y, bottom=clearance_y, left=clearance_x, right=clearance_x, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
                
                    #Adaptive Thresholding // 한번 더 이 작업을 수행하여 숫자의 형태를 분명하게 해준다.
                    num_img_blurred = cv2.GaussianBlur(num_img, ksize=(5,5), sigmaX=0) #노이즈 블러
                    ret, num_thresh = cv2.threshold(num_img_blurred, 127, 255, cv2.THRESH_BINARY)
                    #num_thresh = cv2.adaptiveThreshold(num_img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9) 

                    cv2.imwrite(now_dir+'/number.jpg', num_thresh)
            
                    #KNN 머신러닝데이터로 대조하여 결과 출력
                    FILE_NAME = now_dir + '/number/trained.npz'
                    train, train_labels = load_train_data(FILE_NAME)
                    #KNN
                    test = resize120(num_img)
                    detect_result.append(int(check(test, train, train_labels)))
                number_detect = 'N'
            except:
                detect_result = 'No number'
                number_detect = 'N'
                            
    cap.release()

 
def encodeFrame():
    global thread_lock
    while True:
        # Acquire thread_lock to access the global video_frame object
        with thread_lock:
            global video_frame
            if video_frame is None:
                continue
            return_key, encoded_image = cv2.imencode(".jpg", video_frame)
            if not return_key:
                continue

        # Output image as a byte array
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')

@app.route('/video')
def streamFrames():
    return Response(encodeFrame(), mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route('/', methods=['GET', 'POST'])
def main():
    global number_detect, num_ex, object_detect
    global lower_red1, upper_red1, lower_red2, upper_red2, lower_red3, upper_red3, hsv, color_range
    global threshold_S, threshold_V, threshold
    param = {'threshold_S':threshold_S, 'threshold_V':threshold_V, 'threshold':threshold}
    if request.method == 'POST':
        a = request.form.get('detect')
        b = request.form.get('expansion')
        c = request.form.get('object')
        d = request.form.get('threshold_S')
        e = request.form.get('threshold_V')
        f = request.form.get('threshold')
        
        m_flag = 'N'
        try:
            d = int(float(d))
            if 60 <=d<= 100:
                threshold_S = d
                m_flag = 'Y'
        except:
            pass
        try:
            e = int(float(e))
            if 10<=e<=30:
                threshold_V = e
                m_flag = 'Y'
        except:
            pass
        try:
            f = int(float(f))
            if 127<=f<=250:
                threshold = f
                m_flag = 'Y'
        except:
            pass
        
        if a == 'detect':
            number_detect = 'Y'
        if b=='Y':
            num_ex = 'Y'
        if b=='N':
           num_ex = 'N'
        if c=='OFF':
            object_detect = 'ON'
        if c=='ON':
            object_detect = 'OFF'

        if m_flag == 'Y':
            lower_red1 = np.array([hsv - color_range + 180, threshold_S, threshold_V])
            upper_red1 = np.array([180, 255, 255])
            lower_red2 = np.array([0, threshold_S, threshold_V])
            upper_red2 = np.array([hsv, 255, 255])
            lower_red3 = np.array([hsv, threshold_S, threshold_V])
            upper_red3 = np.array([hsv + color_range, 255, 255])
            
        return redirect('/')
    if num_ex == 'N':
        ex = '보정X'
    elif num_ex == 'Y':
        ex = '보정O'
    
    return render_template('index.html', result = detect_result, ex=ex, object_detect=object_detect, param = param)

@app.route('/result', methods=["GET", "POST"])
def d_result():
    global detect_result
    return jsonify({'result':detect_result})
    
if __name__ == '__main__':
    # Create a thread and attach the method that captures the image frames, to it
    process_thread = threading.Thread(target=captureFrames)
    process_thread.daemon = True

    # Start the thread
    process_thread.start()
    
    # start the Flask Web Application
    # While it can be run on any feasible IP, IP = 0.0.0.0 renders the web app on
    # the host machine's localhost and is discoverable by other machines on the same network 
    app.run(host="0.0.0.0", port="8080")
