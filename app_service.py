# from turtle import shape
from enum import Enum
import threading,cv2
from time import sleep
from unittest import result
from flask import jsonify,stream_with_context,Flask,render_template,Response
from classes.buffer import Buffer
from classes.tensorflow_detection_service import TensorflowDetectionService
from classes.opencv_detection_service import OpencvDetectionService
from classes.pytorch_detection_service import PytorchDetectionService
from classes.stream_reader import StreamSourceEnum, StreamReader
from classes.detection_service import IDetectionService
from classes.background_subtractor_service import BackgroundSubtractorService
from classes.tracking_service import TrackingService
 
from classes.WebcamStream import WebcamStream
from classes.opencv_tensorflow_detection_service import OpencvTensorflowDetectionService
from classes.offline_detector import OfflineDetector
from utils_lib.enums import ClientStreamTypeEnum


class AppService:
    
    stream_reader :StreamReader = None
    detection_service :IDetectionService= None
    background_subtractor_service: BackgroundSubtractorService=None
    tracking_service: TrackingService=None

    file_src   =   "videos/highway2.mp4"
    youtube_url =   "https://www.youtube.com/watch?v=QuUxHIVUoaY"
    webcam_src  =   'http://10.10.23.223:9000/video'
    # video_src = None

    video_src=file_src
    stream_source: StreamSourceEnum=StreamSourceEnum.FILE
    buffering_thread=None


    # youtube_url = "https://www.youtube.com/watch?v=nt3D26lrkho"
    # youtube_url = "https://www.youtube.com/watch?v=QuUxHIVUoaY"
    # youtube_url = "https://www.youtube.com/watch?v=nV2aXhxoJ0Y"
    # youtube_url = "https://www.youtube.com/watch?v=TW3EH4cnFZo"
    # youtube_url = "https://www.youtube.com/watch?v=7y2oOsucOdc"
    # youtube_url = "https://www.youtube.com/watch?v=nt3D26lrkho"
    # youtube_url = "https://www.youtube.com/watch?v=KBsqQez-O4w"
   

    def __init__(self):
      
        print("AppService Starting ...")
        # self.detection_service=TensorflowDetectionService()
        self.detection_service=OpencvDetectionService()
        # self.detection_service=PytorchDetectionService()
        # -----------------
        # self.detection_service=OpencvTensorflowDetectionService()

        #--------------
        # INIT SERVICES
        self.background_subtractor_service=BackgroundSubtractorService()
        self.tracking_service=TrackingService()
        #--------------

        if self.detection_service!=None :
            print( " detection_module loaded succesufuly")
            print( "Service name : ",self.detection_service.service_name())
        else :
            print( " No detection_module To load")
        print("AppService Started.")

        # self.stream_reader=StreamReader(detection_service=self.detection_service,background_subtractor_service=self.background_subtractor_service, stream_source=self.stream_source ,video_src=self.video_src,threshold=self.threshold,nms_threshold=self.nms_threshold) 



    def clean_memory(self):
        print(" START clean_memory ")
        if self.stream_reader:
            self.stream_reader.clean_memory()
        if self.detection_service:
            self.detection_service.clean_memory()
            self.detection_service.init_selected_model()
            # del self.detection_service
        return jsonify(result='clean_memory OK')

    def reset_stream(self):
        self.stream_reader.reset()
        return jsonify('reset stream')
  

    def index(self):
        return render_template('index.html')
 
    def return_stream(self):
        yield from self.stream_reader.read_stream()
        # yield from self.webcam_stream.read_from_camera()

    def stop_stream(self):
        # self.stream_reader.save_records()

        if  not self.stream_reader.stop_reading_from_user_action :
            self.stream_reader.stop_reading_from_user_action=True
            return jsonify(result='stream stoped')
        return jsonify(result='error server in stream stoped')

    def start_stream(self,selected_video):
        selected_video="videos/"+selected_video
        if (selected_video!= self.stream_reader.video_src):
            self.stream_reader.change_video_file(selected_video)
             
        while(True):
            sleep(0.01)
            if self.stream_reader and self.stream_reader.buffer:
                print("SET TO START °°")
                if self.stream_reader.stop_reading_from_user_action :
                    self.stream_reader.stop_reading_from_user_action=False
                    print("SET TO START")
                    return jsonify(result='stream started')
            # return jsonify(result='error server in stream started')

    def start_offline_detection(self):
        # wait for streamer to be created before starting
        print(" START OfflineDetection")
        self.offline_detector=OfflineDetector(self.detection_service,stream_source=self.stream_source ,video_src=self.video_src ) 
        self.offline_detector.threshold= self.stream_reader.threshold  
        self.offline_detector.nms_threshold=self.stream_reader.nms_threshold
        self.offline_detector.start()
        return jsonify(result='OfflineDetector started')

    def get_object_detection_list(self):
        if self.detection_service!=None :
            return jsonify(self.detection_service.get_object_detection_models())
      
    def load_detection_model(self,model=None):
        if self.detection_service!=None :
            self.detection_service.load_model(model=model)
            try:
                # self.detection_service.load_model(model=model)
                return jsonify(result='DONE LOADING SUCCESS')
            except:
                return jsonify(error='ERROR model throw exception')
        return jsonify(result='ERROR model is null')

    def update_threshold_value(self,threshold):
        self.stream_reader.threshold=float(threshold)
        return jsonify(result='threshold updated ')

    def update_nms_threshold_value(self,nms_threshold:float):
        self.stream_reader.nms_threshold=float(nms_threshold)
        return jsonify(result='nmsthreshold updated ')

    def update_background_subtraction_param(self,param,value):
        # self.nms_threshold=float(nms_threshold)
        # self.stream_reader.nms_threshold=self.nms_threshold
        if param=='varThreshold':
            self.background_subtractor_service.background_subtractor.setVarThreshold(int(value))
        if param=='history':
            self.background_subtractor_service.background_subtractor.setHistory(int(value))
        if param=='morphologicalEx':
            self.background_subtractor_service.morphological_ex_iteration=int(value)
        if param=='kernelSize':
            self.background_subtractor_service.kernel_size=int(value)
        if param=='minBoxSize':
            self.background_subtractor_service.min_box_size=int(value)

        return jsonify(result=param+' updated ')

    def main_video_stream(self):
        print("=======> main_video_stream")
        self.stream_reader=StreamReader(detection_service=self.detection_service, stream_source=self.stream_source ,video_src=self.video_src)        
        self.stream_reader.background_subtractor_service=self.background_subtractor_service
        self.stream_reader.tracking_service=self.tracking_service

        self.stream_reader.startBuffering()
        return Response(self.return_stream(),mimetype='text/event-stream')
        # return Response(self.return_stream(),mimetype='multipart/x-mixed-replace; boundary=frame')

    def switch_client_stream(self, stream):
        if  self.stream_reader!=None:
            if stream == 'CNN_DETECTOR':
                self.stream_reader.current_selected_stream= ClientStreamTypeEnum.CNN_DETECTOR
            elif  stream == 'BACKGROUND_SUBTRACTION':
                self.stream_reader.current_selected_stream= ClientStreamTypeEnum.BACKGROUND_SUBTRACTION
            elif stream == 'TRACKING_STREAM':
                self.stream_reader.current_selected_stream= ClientStreamTypeEnum.TRACKING_STREAM

        return jsonify(result=stream)
