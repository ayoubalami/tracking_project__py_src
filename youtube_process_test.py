# from turtle import shape
from concurrent.futures import ThreadPoolExecutor
import math
from threading import Thread
import threading
from time import sleep
from flask import stream_with_context,Flask,render_template,Response
from buffer import Buffer
from stream_reader import StreamReader
from markupsafe import escape

app=Flask(__name__)

# https://github.com/r0oth3x49/Yv-dl/blob/cf52e6f8600bd9cf70c5944a2caade51eadd6a77/pafy/backend_youtube_dl.py#L88
 

def read_stream(stream_reader):
    yield from stream_reader.readStream()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_stream')
def video_stream():

    video_src = "highway1.mp4"
    buffer=Buffer(video_src)
    stream_reader=StreamReader(buffer)
    buffering_thread= threading.Thread(target=buffer.downloadBuffer)
    buffering_thread.start()
    
    
    return Response(read_stream(stream_reader),mimetype='multipart/x-mixed-replace; boundary=frame')



# @app.route('/json')
# def json():
#     # return Response(generate_frames_from_youtube_with_cache(),mimetype='multipart/x-mixed-replace; boundary=frame')
    # return Response(json(),mimetype='multipart/json')

if __name__=="__main__":
    app.run(host='0.0.0.0', port=8000 ,debug=False,threaded=True)

 