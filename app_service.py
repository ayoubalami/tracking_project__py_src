# from turtle import shape
import threading
from enum import Enum
from time import sleep
from unittest import result

import cv2
from flask import (Flask, Response, jsonify, render_template,
                   stream_with_context)

from classes.background_subtractor_service import BackgroundSubtractorService
from classes.buffer import Buffer
from classes.detection_services.detection_service import IDetectionService
from classes.offline_detector import OfflineDetector
from classes.stream_reader import StreamReader, StreamSourceEnum
from classes.tracking_service.tracking_service import TrackingService
from classes.hybrid_tracking_service.hybrid_tracking_service import HybridTrackingService
from classes.stream_processor import StreamProcessor
from classes.tracking_service.offline_tracker import OfflineTracker
from utils_lib.enums import ClientStreamTypeEnum

class AppService:  

    stream_processor :StreamProcessor = None
    detection_service :IDetectionService= None
    background_subtractor_service: BackgroundSubtractorService=None
    tracking_service: TrackingService=None
    hybrid_tracking_service: HybridTrackingService=None
    stream_source: StreamSourceEnum=None
    buffering_thread=None   
    save_detectors_results=False
    host_server='localhost'

    def __init__(self,detection_service:IDetectionService,stream_source:StreamSourceEnum,video_src:str,save_detectors_results:bool,host_server:str):
                
        self.stream_processor=StreamProcessor(self)
        self.detection_service=detection_service
        self.stream_source=stream_source
        self.video_src=video_src
        self.save_detectors_results=save_detectors_results
        self.host_server=host_server
        print("AppService from "+str(self.stream_source) +" Starting ...")
        self.background_subtractor_service=BackgroundSubtractorService()
        self.tracking_service=TrackingService(detection_service=self.detection_service,background_subtractor_service=self.background_subtractor_service)
        self.hybrid_tracking_service=HybridTrackingService(detection_service=self.detection_service,background_subtractor_service=self.background_subtractor_service)

        if self.detection_service!=None :
            print( " detection_module loaded succesufuly")
            print( "Service name : ",self.detection_service.service_name())
            # self.stream_processor.video_stream.detection_service=self.detection_service
        else :
            print( " No detection_module To load")
        print("AppService Started.")

        if stream_source==StreamSourceEnum.RASPBERRY_CAM:
            from classes.raspberry_camera_reader import RaspberryCameraReader
            self.raspberry_camera :RaspberryCameraReader = None
            self.raspberry_camera=RaspberryCameraReader(detection_service=self.detection_service,background_subtractor_service=self.background_subtractor_service,tracking_service=self.tracking_service)
           
    def index(self):
        return render_template('index.html',api_server=self.host_server,stream_source=self.stream_source.name)
         
    def stop_stream(self):
        self.stream_processor.video_stream.stop()
        return jsonify(result='error server in stream stoped')
           
    def reset_stream(self):
        self.stream_processor.video_stream.reset()
        return jsonify(result='stream reset')

    def start_stream(self,selected_video):
        if (self.stream_processor.video_stream.video_file!=selected_video):
            self.stream_processor.video_stream.video_file=selected_video
        self.stream_processor.video_stream.start()
        return jsonify(result='stream started')

    def one_next_frame(self):
        print("GET NEXT FRAME START °°")
        self.stream_processor.video_stream.one_next_frame=True
        return jsonify(result='error getting  NEXT FRAME ')
    
    def get_object_detection_list(self):
        if self.detection_service!=None :
            return jsonify(self.detection_service.get_object_detection_models())
      
    def load_detection_model(self,model=None):
        if self.detection_service!=None :
            self.detection_service.load_model(model=model)
            try:
                return jsonify(result='DONE LOADING SUCCESS')
            except:
                return jsonify(error='ERROR model throw exception')
        return jsonify(result='ERROR model is null')

    def update_cnn_detector_param(self,param,value):
        if param=='networkInputSize':
            self.detection_service.network_input_size=int(value)
        elif param=='threshold':
            self.detection_service.threshold=float(value)
        elif param=='nmsThreshold':
            self.detection_service.nms_threshold=float(value)
        return jsonify(result=param+' updated ')

    def update_background_subtraction_param(self,param,value):
        if param=='varThreshold':
            self.background_subtractor_service.background_subtractor.setVarThreshold(int(value))
        if param=='history':
            self.background_subtractor_service.background_subtractor.setHistory(int(value))
        if param=='blurKernelSize':
            self.background_subtractor_service.blur_kernel_size=int(value)
        if param=='morphologicalEx':
            self.background_subtractor_service.morphological_ex_iteration=int(value)
        if param=='morphologicalKernelSize':
            self.background_subtractor_service.morphological_kernel_size=int(value)
        if param=='minBoxSize':
            self.background_subtractor_service.min_box_size=int(value)
        return jsonify(result=param+' updated ')

    def main_video_stream(self): 
        print("=======> main_video_stream")
        return Response(self.stream_processor.return_stream(),mimetype='text/event-stream')

    def update_tracking_param_value(self,param,value):
        if self.tracking_service:
            if param == 'maxCosDistance':
                self.tracking_service.threshold_feature_distance= float(value)
            if param =='maxDistance':
                self.tracking_service.tracker.max_iou_distance= float(value)
        return jsonify(result=value)

    def rotate_servo_motor(self,axis,value):
        if self.stream_source==StreamSourceEnum.RASPBERRY_CAM:
            self.raspberry_camera.rotateServoMotor(axis=axis,angle=int(value),speed=0.005)
        return jsonify(result=value)

    def update_raspberry_camera_zoom(self,zoom):
        if self.stream_source==StreamSourceEnum.RASPBERRY_CAM:
            self.raspberry_camera.zoom= float(zoom)
        return jsonify(result=zoom)

    def update_tracked_coordinates(self,x,y):
        if  self.stream_source==StreamSourceEnum.RASPBERRY_CAM and self.tracking_service:
            tracked_object=self.tracking_service.returnSelectedTrackedObject((float(x),float(y)))
            
            if tracked_object:
                print("AN OBJECT TO TRACK IS FOUNDED : " +str(tracked_object.track_id)+" - "+str(tracked_object.to_tlwh())) 
                self.raspberry_camera.moveServoMotorToCoordinates(tracked_object)

            else:
                self.raspberry_camera.terminate_tracking=True
                print("NO OBJECT FOUNDED FROM APP_SERVICE")
        return jsonify(result=x)

    def get_class_labels(self):
        classLables=[]
        classFile ="coco.names" 
        with open(classFile, 'r') as f:
            lables = f.read().splitlines() 
            i=0
            for lable in lables:
                classLables.append({'id':i,'label': lable })
                i+=1
        return jsonify(classLables)

    def set_selected_classes( self,idx):
        idx=list(map(int, idx.split(",")))
        if not (len(idx)>0 and idx[0]==-1):
            self.detection_service.allowed_classes=idx
        else:
            self.detection_service.allowed_classes=[]
        return jsonify(result=idx)

    def change_video_file(self,video_file):
        self.stream_processor.video_stream.change_video_file(video_file)
        return jsonify(result="video changed")
