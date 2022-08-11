from flask import Flask, Response
import cv2
import numpy as np
import threading 
import requests
import os
import socket

img_w = 640
img_h = 480

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
        break
    except:
        # 경고문은 추후 LCD로 출력되도록
        if i_flag == 'Y':
            print('No internet')
            i_flag = 'N'
            
# 리눅스에서 os.popen('hostname -I').read().strip() (내부)
in_ip = socket.gethostbyname(socket.gethostname())
print(in_ip, ex_ip)
# Flask server 선언
app = Flask(__name__)

# main loop code 
def captureFrames():
    global video_frame, thread_lock
    #아래 코드는 윈도우에서 쓸때로 리눅스에선 cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_h)
    while True:
        ret, frame = cap.read()
        
        if not ret:
            #오류 페이지 생성
            return '<h1>Error:</h1> <p>Camera is not opened...</p>'
        
        with thread_lock:
            video_frame = frame.copy()              
    
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

@app.route('/')
def streamFrames():
    return Response(encodeFrame(), mimetype = "multipart/x-mixed-replace; boundary=frame")

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