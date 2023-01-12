# from turtle import shape
# ps aux
from threading import Thread
import threading,os
from time import sleep,time
from flask import jsonify,stream_with_context,Flask,render_template,Response
from classes.buffer import Buffer
from classes.tensorflow_detection_service import TensorflowDetectionService
from classes.stream_reader import StreamSourceEnum, StreamReader
from classes.detection_service import IDetectionService
from app_service import AppService
from flask_cors import CORS
import sys,argparse

from classes.tensorflow_detection_service import TensorflowDetectionService
from classes.opencv_detection_service import OpencvDetectionService
from classes.onnx_detection_service import OnnxDetectionService
from classes.yolov5_detection_service import Yolov5DetectionService
from classes.trash.yolov8_detection_service import Yolov8DetectionService

def pars_args():
    file_src   =   "videos/highway2.mp4"
    # youtube_url =   "https://www.youtube.com/watch?v=QuUxHIVUoaY"
    webcam_src  =   'http://192.168.43.1:9000/video'
#    webcam_src=0
    stream_source: StreamSourceEnum=StreamSourceEnum.FILE
    parser = argparse.ArgumentParser()
    save_detectors_results=False
    host_server='localhost'
    
    parser.add_argument("-hs", "--host_server", help = "host_server host remote raspberry of local pc")
    parser.add_argument("-ss", "--stream_source", help = "Select stream source FILE, WEBCAM")
    parser.add_argument("-ds", "--detection_service", help = "Select detection service OPENCV, PYTORCH, TENSORFLOW")
    parser.add_argument("-rr", "--save_detectors_results", help = "save_detectors_results inference fps to results.csv")
    parser.add_argument("-cam_stream", "--webcam", dest='webcam', help = "",action="store")

    args = parser.parse_args()
    if args:
        detection_service:IDetectionService=None
        if args.detection_service:
            if args.detection_service in( 'OPENCV' ,'opencv') :
                detection_service=OpencvDetectionService()
            elif args.detection_service in( 'ONNX' ,'onnx')  :
                detection_service=OnnxDetectionService()
            elif args.detection_service in( 'TENSORFLOW' ,'tf') :
                detection_service=TensorflowDetectionService()
            elif args.detection_service in( 'YOLOV5' ,'yv5') :
                detection_service=Yolov5DetectionService()
            # elif args.detection_service in( 'YOLOV8' ,'yv8') :
            #     detection_service=Yolov8DetectionService()

        else:
            detection_service=OpencvDetectionService()

        if args.stream_source:
            if args.stream_source in( 'FILE' ,'f') :
                stream_source=StreamSourceEnum.FILE
                video_src=file_src
            elif args.stream_source in( 'WEBCAM' ,'w')  :
                stream_source=StreamSourceEnum.WEBCAM
                video_src=webcam_src
                if args.webcam :
                    if args.webcam in( 'r' ,'rasp','raspberry')  :
                        stream_source=StreamSourceEnum.RASPBERRY_CAM
                    else:
                        video_src= args.webcam
                    
                    print('read from WEBCAM'+ video_src)
        else:
            stream_source=StreamSourceEnum.FILE
            video_src=file_src

        if args.save_detectors_results:
            save_detectors_results=True

        if args.host_server:
            if args.host_server in( 'r' ,'rasp','raspberry','raspberrypi') :
                host_server='raspberrypi.local'
            else :
                host_server=args.host_server

            
    return detection_service,stream_source,video_src,save_detectors_results,host_server

app=Flask(__name__)
CORS(app)

detection_service,stream_source,video_src,save_detectors_results,host_server=pars_args()
app_service=AppService(detection_service=detection_service,stream_source=stream_source,video_src=video_src,save_detectors_results=save_detectors_results,host_server=host_server)


# def read_stream():
#     print("read_stream")
#     yield from app_service.read_stream()
    # pass

@app.route('/')
def index():
    return app_service.index()

@app.route('/main_video_stream')
def main_video_stream():
    return app_service.main_video_stream()


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

@app.route('/start_stream/<selected_video>', methods = ['POST'])
def start_stream(selected_video):
    return app_service.start_stream(selected_video)

@app.route('/start_offline_detection', methods = ['POST'])
def start_offline_detection():
    return app_service.start_offline_detection()

@app.route('/get_object_detection_list', methods = ['GET'])
def get_object_detection_list():
    return app_service.get_object_detection_list()
    

@app.route('/models/load/<model>', methods = ['POST'])
def load_detection_model(model):
    return app_service.load_detection_model(model=model)

@app.route('/models/update_threshold/<threshold>', methods = ['POST'])
def update_threshold_value(threshold):
    return app_service.update_threshold_value(threshold=threshold)

@app.route('/models/update_nms_threshold/<nms_threshold>', methods = ['POST'])
def update_nms_threshold_value(nms_threshold):
    return app_service.update_nms_threshold_value(nms_threshold=nms_threshold)

@app.route('/models/update_background_subtraction_param/<param>/<value>', methods = ['POST'])
def update_background_subtraction_param(param,value):
    return app_service.update_background_subtraction_param(param=param,value=value)


@app.route('/switch_client_stream/<stream>', methods = ['POST'])
def switch_client_stream(stream):
    return app_service.switch_client_stream(stream=stream)

@app.route('/stream/reset', methods = ['POST'])
def reset_stream():
    return app_service.reset_stream() 
    # return app_service.clean_memory() 


if __name__=="__main__":
    app.run(host='0.0.0.0', port=8000 ,debug=False,threaded=True)

 

# youtube_url = "https://www.youtube.com/watch?v=nt3D26lrkho"
# youtube_url = "https://www.youtube.com/watch?v=QuUxHIVUoaY"
# youtube_url = "https://www.youtube.com/watch?v=nV2aXhxoJ0Y"
# youtube_url = "https://www.youtube.com/watch?v=TW3EH4cnFZo"
# youtube_url = "https://www.youtube.com/watch?v=7y2oOsucOdc"
# youtube_url = "https://www.youtube.com/watch?v=nt3D26lrkho"
# youtube_url = "https://www.youtube.com/watch?v=KBsqQez-O4w"
