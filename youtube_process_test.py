# from turtle import shape
from concurrent.futures import ThreadPoolExecutor
import math
from threading import Thread
import threading
from time import sleep,time
from flask import jsonify,stream_with_context,Flask,render_template,Response
from buffer import Buffer
from stream_reader import StreamReader
from markupsafe import escape

app=Flask(__name__)

# https://github.com/r0oth3x49/Yv-dl/blob/cf52e6f8600bd9cf70c5944a2caade51eadd6a77/pafy/backend_youtube_dl.py#L88


global buffering_thread
buffering_thread=None

def read_stream(stream_reader):
    yield from stream_reader.readStream()
    # pass

@app.route('/')
def index():
    return render_template('index.html')


def kill_buffring_thread():
    # global buffering_thread
    print( ' befor join thread ') 
    global buffer
    buffer.stop_buffring_event.set()
    print('thread killed : buffering_thread')  


@app.route('/video_stream')
def video_stream():
    global stream_reader
    global buffer
    global buffering_thread

    video_src = "highway4.mp4"
    buffer=Buffer(video_src=video_src)
    
    # youtube_url = "https://www.youtube.com/watch?v=KBsqQez-O4w"
    # youtube_url = "https://www.youtube.com/watch?v=QuUxHIVUoaY"
    # youtube_url = "https://www.youtube.com/watch?v=nt3D26lrkho"
    # buffer=Buffer(youtube_url=youtube_url)

    stream_reader=StreamReader(buffer)
    buffering_thread= threading.Thread(target=buffer.downloadBuffer)
    buffering_thread.start()
    return Response(read_stream(stream_reader),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/current_time', methods = ['GET'])
def timer():
    global stream_reader
    if stream_reader != None :
        return jsonify(result=stream_reader.current_time)
    else :
        return jsonify(result=1)


@app.route('/video_duration', methods = ['GET'])
def videoDuration():
    return jsonify(result=buffer.video_duration)


@app.route('/clean_memory', methods = ['POST'])
def clean_memory():
    # global buffer
    # buffer.clean_memory()
    # print('START CLEANING...')  
    kill_buffring_thread()
    # buffer.clean_memory()
    return jsonify(result='DONE')


if __name__=="__main__":
    app.run(host='0.0.0.0', port=8000 ,debug=False,threaded=True)

