# from turtle import shape
# ps aux
from threading import Thread
import threading,os
from time import sleep,time
from flask import jsonify,stream_with_context,Flask,render_template,Response
from classes.buffer import Buffer
from classes.stream_reader import StreamSourceEnum, StreamReader
from classes.detection_services.detection_service import IDetectionService
from app_service import AppService
from flask_cors import CORS
import sys,argparse
from classes.detection_services.tensorflow_detection_service import TensorflowDetectionService
from classes.detection_services.tensorflow_lite_detection_service import TensorflowLiteDetectionService

from classes.detection_services.opencv_detection_service import OpencvDetectionService
from classes.detection_services.onnx_detection_service import OnnxDetectionService
from classes.detection_services.yolov5_detection_service import Yolov5DetectionService
from classes.detection_services.yolov6_detection_service import Yolov6DetectionService
from classes.detection_services.yolov8_detection_service import Yolov8DetectionService

def pars_args():
    file_src   =   "videos/highway2.mp4"
    # webcam_src  =   'http://192.168.43.1:9000/video'
    local_webcam_src  =   'http://10.10.23.223:9000/video'
#    webcam_src=0
    stream_source: StreamSourceEnum=StreamSourceEnum.FILE
    parser = argparse.ArgumentParser()
    save_detectors_results=False
    host_server='localhost'
    # host_server='10.10.23.116'
    
    parser.add_argument("-hs", "--host_server", help = "host_server host remote raspberry of local pc")
    parser.add_argument("-ss", "--stream_source", help = "Select stream source FILE, REMOTE_WEBCAM, RASPBERRY_CAM")
    parser.add_argument("-ds", "--detection_service", help = "Select detection service OPENCV, PYTORCH, TENSORFLOW")
    parser.add_argument("-rr", "--save_detectors_results",  action="store_const", const=True, default=False, help = "save_detectors_results inference fps to results.csv")
    parser.add_argument("-ws", "--webcam", help = "webcam ip server ")

    args = parser.parse_args()
    video_src=file_src

    if args:
        detection_service:IDetectionService=None
        if args.detection_service:
            if args.detection_service in( 'OPENCV' ,'opencv') :
                detection_service=OpencvDetectionService()
            elif args.detection_service in( 'ONNX' ,'onnx')  :
                detection_service=OnnxDetectionService()
            elif args.detection_service in( 'TENSORFLOW' ,'tf','tensorflow') :
                detection_service=TensorflowDetectionService()
            elif args.detection_service in( 'TENSORFLOW LITE' ,'tflite','tensorflow-lite') :
                detection_service=TensorflowLiteDetectionService()
            elif args.detection_service in( 'YOLOv5' ,'yv5') :
                detection_service=Yolov5DetectionService()
            elif args.detection_service in( 'YOLOv6' ,'yv6') :
                detection_service=Yolov6DetectionService()
            elif args.detection_service in( 'YOLOv8' ,'yv8') :
                detection_service=Yolov8DetectionService()
            else:
                print("PARAM INCORRECT. LOADING DEFAULT DETECTION SERVICE ....")
                detection_service=OpencvDetectionService()
        else:
            detection_service=OpencvDetectionService()

        if args.stream_source:
            if args.stream_source in( 'FILE' ,'f') :
                stream_source=StreamSourceEnum.FILE
                video_src=file_src
            elif args.stream_source in( 'WEBCAM' ,'w')  :
                stream_source=StreamSourceEnum.WEBCAM
                video_src=local_webcam_src
            elif args.stream_source in( 'RASPBERRY' ,'r')  :
                stream_source=StreamSourceEnum.RASPBERRY_CAM
                # video_src=webcam_src
                # if args.webcam :
                #     if args.webcam in( 'r' ,'rasp','raspberry')  :
                #         stream_source=StreamSourceEnum.RASPBERRY_CAM
                #     else:
                video_src= 'RASPBERRY_CAM'
                    
            print('read from '+ str(stream_source))
        else:
            stream_source=StreamSourceEnum.FILE
            video_src=file_src

        if args.webcam :
            stream_source=StreamSourceEnum.WEBCAM
            if str(args.webcam)=='-':
                video_src=local_webcam_src
            else:
                video_src=str(args.webcam)

        if args.save_detectors_results:
            save_detectors_results=True

        if args.host_server:
            if args.host_server in( 'r' ,'rasp','raspberry','raspberrypi') :
                host_server='raspberrypi.local'
                # host_server='192.168.43.186'
            else :
                host_server=args.host_server

    return detection_service,stream_source,video_src,save_detectors_results,host_server

app=Flask(__name__)
CORS(app)

# detection_service,stream_source,video_src,save_detectors_results,host_server=
app_service=AppService(*pars_args())

@app.route('/')
def index():
    return app_service.index()

@app.route('/main_video_stream')
def main_video_stream():
    response=app_service.main_video_stream()
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# @app.route('/current_time', methods = ['GET'])
# def timer():
#     return app_service.timer()

# @app.route('/video_duration', methods = ['GET'])
# def video_duration():
#     return app_service.video_duration()

# @app.route('/clean_memory', methods = ['POST'])
# def clean_memory():
#     return app_service.clean_memory()

@app.route('/stop_stream', methods = ['POST'])
def stop_stream():
    return app_service.stop_stream()

# <video_resolution_ratio>/<selected_video>
# def start_stream( selected_video,video_resolution_ratio):

@app.route('/start_stream/<selected_video>', methods = ['POST'])
def start_stream(selected_video):
    return app_service.start_stream(selected_video)

@app.route('/get_next_frame', methods = ['POST'])
def get_next_frame():
    return app_service.one_next_frame()

# @app.route('/start_offline_detection/<selected_video>', methods = ['POST'])
# def start_offline_detection(selected_video):
#     return app_service.start_offline_detection(selected_video)

# @app.route('/start_offline_tracking/<selected_video>', methods = ['POST'])
# def start_offline_tracking(selected_video):
#     return app_service.start_offline_tracking(selected_video)

@app.route('/get_object_detection_list', methods = ['GET'])
def get_object_detection_list():
    return app_service.get_object_detection_list()
    
@app.route('/models/load/<model>', methods = ['POST'])
def load_detection_model(model):
    return app_service.load_detection_model(model=model)

# @app.route('/models/update_threshold/<threshold>', methods = ['POST'])
# def update_threshold_value(threshold):
#     return app_service.update_threshold_value(threshold=threshold)

# @app.route('/models/update_nms_threshold/<nms_threshold>', methods = ['POST'])
# def update_nms_threshold_value(nms_threshold):
#     return app_service.update_nms_threshold_value(nms_threshold=nms_threshold)

@app.route('/models/update_cnn_detector_param/<param>/<value>', methods = ['POST'])
def update_cnn_detector_param(param,value):
    return app_service.update_cnn_detector_param(param=param,value=value)

@app.route('/models/update_background_subtraction_param/<param>/<value>', methods = ['POST'])
def update_background_subtraction_param(param,value):
    return app_service.update_background_subtraction_param(param=param,value=value)

@app.route('/switch_client_stream/<stream>', methods = ['POST'])
def switch_client_stream(stream):
    return app_service.switch_client_stream(stream=stream)

@app.route('/reset_stream', methods = ['POST'])
def reset_stream():
    return app_service.reset_stream() 

@app.route('/track_with/<param>', methods = ['POST'])
def track_with(param):
    return app_service.track_with(param=param) 

@app.route('/activate_stream_simulation/<value>', methods = ['POST'])
def activate_stream_simulation(value):
    return app_service.activate_stream_simulation(value=value) 

# @app.route('/show_missing_tracks/<value>', methods = ['POST'])
# def show_missing_tracks(value):
#     return app_service.show_missing_tracks(value=value) 

# @app.route('/use_cnn_feature_extraction_on_tracking/<value>', methods = ['POST'])
# def use_cnn_feature_extraction_on_tracking(value):
#     return app_service.use_cnn_feature_extraction_on_tracking(value=value) 

# @app.route('/activate_detection/<value>', methods = ['POST'])
# def activate_detection_for_tracking(value):
#     return app_service.activate_detection_for_tracking(value=value) 

# @app.route('/update_tracking_param/<param>/<value>', methods = ['POST'])
# def update_tracking_param_value(param,value):
#     return app_service.update_tracking_param_value(param=param,value=value) 

# @app.route('/rotate_servo_motor/<axis>/<value>', methods = ['POST'])
# def rotate_servo_motor(axis,value):
#     return app_service.rotate_servo_motor(axis=axis,value=value) 

# @app.route('/update_raspberry_camera_zoom/<zoom>', methods = ['POST'])
# def update_raspberry_camera_zoom(zoom):
#     return app_service.update_raspberry_camera_zoom(zoom=zoom) 

# @app.route('/update_tracked_coordinates/<x>/<y>', methods = ['POST'])
# def update_tracked_coordinates(x,y):
    # return app_service.update_tracked_coordinates(x,y) 

@app.route('/get_class_labels', methods = ['POST'])
def get_class_labels( ):
    return app_service.get_class_labels( ) 

@app.route('/set_selected_classes/<idx>', methods = ['POST'])
def set_selected_classes( idx):
    return app_service.set_selected_classes(idx ) 

@app.route('/change_video_file/<video_file>', methods = ['POST'])
def change_video_file( video_file):
    return app_service.change_video_file(video_file ) 

@app.route('/BS_set_video_resolution/<video_resolution_ratio>', methods = ['POST'])
def BS_set_video_resolution( video_resolution_ratio):
    return app_service.BS_set_video_resolution(video_resolution_ratio ) 

@app.route('/CNN_set_video_resolution/<video_resolution_ratio>', methods = ['POST'])
def CNN_set_video_resolution( video_resolution_ratio):
    return app_service.CNN_set_video_resolution(video_resolution_ratio ) 

@app.route('/set_video_starting_second/<second>', methods = ['POST'])
def set_video_starting_second( second):
    print(f" set starting_second to : {second}")
    return app_service.set_video_starting_second( int(second) ) 

if __name__=="__main__":
    app.run(host='0.0.0.0', port=7070 ,debug=False,threaded=True)