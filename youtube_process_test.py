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
 

def read_stream(stream_reader):
    yield from stream_reader.readStream()
    # pass

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/video_stream')
def video_stream():
    global stream_reader
    global buffer

    video_src = "highway2.mp4"
   
    buffer=Buffer(video_src)
    stream_reader=StreamReader(buffer)
    buffering_thread= threading.Thread(target=buffer.downloadBuffer)
    buffering_thread.start()
    return Response(read_stream(stream_reader),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/current_time', methods = ['GET'])
def timer():
    global stream_reader
    if stream_reader != None :
        return jsonify(result=stream_reader.current_time)
    else    :
        return jsonify(result=1)

@app.route('/video_duration', methods = ['GET'])
def videoDuration():
    return jsonify(result=buffer.video_duration)


if __name__=="__main__":
    app.run(host='0.0.0.0', port=8000 ,debug=False,threaded=True)

 