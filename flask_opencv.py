import cv2
import time
from flask import Response, Flask, render_template, stream_with_context
import numpy as np
import datetime
import math
import time
import threading

#initial value
img_w = 640
img_h = 480

#declare Flask Server
app = Flask(__name__)

def captureFrames():
    global video_frame, thread_lock

    # Video capturing
    cap = cv2.VideoCapture(0)

    # Set Video Size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_h)
    
    while True and cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            '''camera is not detected'''
            break
        
        #filter

        #encoding
        return_key, encoded_image = cv2.imencode(".jpg", frame)
        if not return_key:
            continue
        
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')
            
@app.route('/video')
def streamFrames():
    return Response(captureFrames(), mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route('/')
def index():
    return render_template('index.html')

# check to see if this is the main thread of execution
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