from typing import Iterable
from py import process
from process import Process
from flask import Flask,render_template,Response,url_for
import cv2
import numpy as np

app = Flask(__name__)
camera=cv2.VideoCapture(0)
frame = ''
process = Process()
hr = 0

def generate_frames():
    while True:
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            hr, spo2 = process.process(frame) 
            cv2.putText(frame, '{:.0f}bpm'.format(hr), (30, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (255, 0, 0), 2)
            cv2.putText(frame, '{:.0f}%'.format(spo2), (30, 90), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 0, 255), 2)


            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
   

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)