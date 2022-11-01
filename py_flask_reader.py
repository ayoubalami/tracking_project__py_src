# from turtle import shape
from threading import Thread
import threading,os
from time import sleep,time
from flask import jsonify,stream_with_context,Flask,render_template,Response
from classes.buffer import Buffer
from classes.tensorflow_detection_service import TensorflowDetectionService
from classes.stream_reader import StreamReader
from classes.detection_service import IDetectionService
# from classes.zoo_tensorflow_detection_service import ZooTensorflowDetectionService
 
app=Flask(__name__)

# https://github.com/r0oth3x49/Yv-dl/blob/cf52e6f8600bd9cf70c5944a2caade51eadd6a77/pafy/backend_youtube_dl.py#L88
# youtube_url = "https://www.youtube.com/watch?v=KBsqQez-O4w"
# youtube_url = "https://www.youtube.com/watch?v=QuUxHIVUoaY"
# youtube_url = "https://www.youtube.com/watch?v=nt3D26lrkho"

# buffer=Buffer(youtube_url=youtube_url)
stream_reader :StreamReader = None
buffer :Buffer= None
detection_service :IDetectionService= None

print("DONE")
def read_stream():
    global stream_reader    
    yield from stream_reader.readStream()
    # pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_stream')
def video_stream():
    global stream_reader
    global buffer
    global detection_service

    detection_service=TensorflowDetectionService()
    # detection_service=TensorflowDetectionService()
    print( " detection_module loaded succesufuly")
    print( "Service name : ",detection_service.service_name())
    print( "Model name : ", detection_service.model_name())
     
    # youtube_url = "https://www.youtube.com/watch?v=QuUxHIVUoaY"
    # buffer=Buffer(youtube_url=youtube_url)

    video_src = "videos/highway2.mp4"
    buffer=Buffer(video_src=video_src)

    stream_reader=StreamReader(buffer,detection_service)

    buffering_thread= threading.Thread(target=buffer.downloadBuffer)
    buffering_thread.start()

    return Response(read_stream(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/current_time', methods = ['GET'])
def timer():
    global stream_reader

    if stream_reader != None :
        return jsonify(result=stream_reader.current_time)
    else :
        return jsonify(result=1)

@app.route('/video_duration', methods = ['GET'])
def videoDuration():
    # global buffer
    return jsonify(result=buffer.video_duration)

@app.route('/clean_memory', methods = ['POST'])
def clean_memory():
    # global buffer
    if buffer :
        buffer.stop_buffring_event.set()
        return jsonify(result='DONE')
    return jsonify(result='clean_memory ERROR')

@app.route('/stop_stream', methods = ['POST'])
def stopStream():
    global stream_reader
    if buffer and not stream_reader.stop_reading_from_user_action :
        stream_reader.stop_reading_from_user_action=True
        return jsonify(result='stream stoped')
    return jsonify(result='error server in stream stoped')

@app.route('/start_stream', methods = ['POST'])
def startStream():
    global stream_reader
    if buffer and stream_reader.stop_reading_from_user_action :
        stream_reader.stop_reading_from_user_action=False
        return jsonify(result='stream started')
    return jsonify(result='error server in stream started')

    



if __name__=="__main__":
    app.run(host='0.0.0.0', port=8000 ,debug=False,threaded=True)

