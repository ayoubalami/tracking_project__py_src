# from turtle import shape
from concurrent.futures import thread
from difflib import Match
import math
import threading
from flask import Flask,render_template,Response
import cv2,time
import pafy
import pyshine as ps

app=Flask(__name__)

# https://github.com/r0oth3x49/Yv-dl/blob/cf52e6f8600bd9cf70c5944a2caade51eadd6a77/pafy/backend_youtube_dl.py#L88

cache_frames=[]
def generate_frames_from_youtube_with_cache():
    # video_src='https://www.youtube.com/watch?v=KBsqQez-O4w'
    # urlPafy = pafy.new(video_src)
    # videoplay = urlPafy.getbest(preftype="any")
    # streams = urlPafy.allstreams
    # # print(streams)
    
    # videoplay = urlPafy.getbestvideo(preftype="mp4")
    # priority=[480,360,720]
    # indexes = [i for i in range(len(streams)) if priority[0] == streams[i]._dimensions[1] and streams[i]._mediatype == 'video' ]
    # if len(indexes)>0:
    #     videoplay = streams[indexes[0]]
    # else:
    #     indexes = [i for i in range(len(streams)) if priority[1] == streams[i]._dimensions[1] and streams[i]._mediatype == 'video' ]
    #     if len(indexes)>0:
    #         videoplay = streams[indexes[0]]
    #     else:      
    #         indexes = [i for i in range(len(streams)) if priority[2] == streams[i]._dimensions[1] and streams[i]._mediatype == 'video' ]
    #         videoplay = streams[indexes[0]]

    # cap = cv2.VideoCapture(videoplay.url)
    # cache_frames=[]
    
    # 
    global cache_frames
    # loadBuffer()
    buffering_thread=threading.Thread(target=loadBuffer)
    buffering_thread.start()
    print("conti....")
    # delay=0.03
    # target_delay=0.0333333
    # print(cache_frames)

    iterator=0
    while True :
        print (len(cache_frames))
        time.sleep(0.01)
    # while iterator<len(cache_frames)  :
    #     t1= time.time()
    #     ret,buffer=cv2.imencode('.jpg',cache_frames[iterator])
    #     frame=buffer.tobytes()
    #     print("yeild frame ...")
    #     # time.sleep(delay)
    #     yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    #     iterator=iterator+1
    #     delay=(time.time()-t1)
    #     # print (target_delay)
    #     # print (delay)
    #     delay=target_delay-delay
    #     if delay>0:
    #         time.sleep(delay)

    print("end buffer read !")


def loadBuffer_() :
    global cache_frames
    cap = cv2.VideoCapture("highway1.mp4")
    print(cap.get(cv2.CAP_PROP_BUFFERSIZE))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)

    cache_size=100
    iterator=0

    # cache_frames=[]
    while iterator<cache_size  :
        success, frame = cap.read()
        cache_frames.append(frame)
        iterator=iterator+1
    print("buffer read !")
    iterator=0
    # return cache_frames

def loadBuffer() :
    while(True):
        loadBuffer_()
   

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_stream')
def video_stream():
    return Response(generate_frames_from_youtube_with_cache(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(host='0.0.0.0', port=8000 ,debug=False,threaded=True)

 