# from turtle import shape
from threading import Thread
import threading,os
from time import sleep,time
from flask import jsonify,stream_with_context,Flask,render_template,Response
from classes.buffer import Buffer
from classes.tensorflow_detection_service import TensorflowDetectionService
from classes.stream_reader import StreamReader
from classes.detection_service import IDetectionService
from app_service import AppService

app=Flask(__name__)
app_service=AppService()

# def read_stream():
#     print("read_stream")
#     yield from app_service.read_stream()
    # pass

@app.route('/')
def index():
    return app_service.index()

@app.route('/video_stream')
def video_stream():
    return app_service.video_stream()

@app.route('/current_time', methods = ['GET'])
def timer():
    return app_service.timer()

@app.route('/video_duration', methods = ['GET'])
def video_duration():
    return app_service.video_duration()

@app.route('/clean_memory', methods = ['POST'])
def clean_memory():
    return app_service.clean_memory()

@app.route('/stop_stream', methods = ['POST'])
def stop_stream():
    return app_service.stop_stream()

@app.route('/start_stream', methods = ['POST'])
def start_stream():
    return app_service.start_stream()

@app.route('/get_object_detection_list', methods = ['GET'])
def get_object_detection_list():
    return app_service.get_object_detection_list()
    

@app.route('/models/load/<model>', methods = ['POST'])
def load_detection_model(model):
    return app_service.load_detection_model(model=model)


@app.route('/stream/reset', methods = ['POST'])
def reset_stream():
    return app_service.reset_stream() 
    # return app_service.clean_memory() 


if __name__=="__main__":
    app.run(host='0.0.0.0', port=8000 ,debug=False,threaded=True)

