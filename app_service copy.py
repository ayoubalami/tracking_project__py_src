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

from classes.tracking_service.offline_tracker import OfflineTracker
from utils_lib.enums import ClientStreamTypeEnum

class AppService:  

    stream_reader :StreamReader = None
    detection_service :IDetectionService= None
    background_subtractor_service: BackgroundSubtractorService=None
    tracking_service: TrackingService=None
    hybrid_tracking_service: HybridTrackingService=None
    stream_source: StreamSourceEnum=None
    buffering_thread=None   
    save_detectors_results=False
    host_server='localhost'

    def __init__(self,detection_service:IDetectionService,stream_source:StreamSourceEnum,video_src:str,save_detectors_results:bool,host_server:str):
        
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
        else :
            print( " No detection_module To load")
        print("AppService Started.")

        if stream_source==StreamSourceEnum.RASPBERRY_CAM:
            from classes.raspberry_camera_reader import RaspberryCameraReader
            self.raspberry_camera :RaspberryCameraReader = None
            self.raspberry_camera=RaspberryCameraReader(detection_service=self.detection_service,background_subtractor_service=self.background_subtractor_service,tracking_service=self.tracking_service)
           

        # self.stream_reader=StreamReader(detection_service=self.detection_service, stream_source=self.stream_source ,video_src=self.video_src)        

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
        if self.stream_reader:
            self.stream_reader.reset()
        if self.tracking_service:
            self.tracking_service.reset()
        if self.hybrid_tracking_service:
            self.hybrid_tracking_service.reset()
        return jsonify('reset stream')
  
    def reset_buffer_starting_second(self,starting_second=0):
        
        # self.stop_stream()
        if self.stream_reader:
            self.stream_reader.reset(starting_second=starting_second)
        if self.tracking_service:
            self.tracking_service.reset()
        if self.hybrid_tracking_service:
            self.hybrid_tracking_service.reset()

        # self.start_stream()
        return jsonify('reset stream')


    def index(self):
        return render_template('index.html',api_server=self.host_server,stream_source=self.stream_source.name)
 
    def return_stream(self):
        yield from self.stream_reader.read_stream()
        # yield from self.webcam_stream.read_from_camera()

    def stop_stream(self):
        if self.stream_source==StreamSourceEnum.RASPBERRY_CAM:
            if self.raspberry_camera.start_reading_action  :
                self.raspberry_camera.start_reading_action=False
                print("SET CAMERA MODULE STOP")
                return jsonify(result='rasp stream stoped')
        else:
            if self.save_detectors_results:
                self.stream_reader.save_records()
            if  not self.stream_reader.stop_reading_from_user_action :
                self.stream_reader.stop_reading_from_user_action=True
                return jsonify(result='stream stoped')
            return jsonify(result='error server in stream stoped')

    def start_stream(self,selected_video,video_resolution_ratio):
        if self.stream_source==StreamSourceEnum.RASPBERRY_CAM:
            if self.raspberry_camera.start_reading_action ==False:
                self.raspberry_camera.start_reading_action=True
                print("SET CAMERA MODULE START")
                return jsonify(result='rasp stream started')
        else:
            selected_video="videos/"+selected_video
            video_resolution_ratio=float(video_resolution_ratio)/100
            if video_resolution_ratio>1:
                video_resolution_ratio=1

            self.background_subtractor_service.video_resolution_ratio = video_resolution_ratio

            if (selected_video!= self.stream_reader.video_src):
                self.video_src=selected_video
                self.stream_reader.change_video_file(selected_video)
                if self.tracking_service:
                    self.tracking_service.reset()
               

            # self.stream_reader.buffer.set_buffer_starting_frame(50)

            # while(True):
            #     sleep(0.01)
                # if self.stream_reader and (self.stream_reader.buffer or self.stream_source == StreamSourceEnum.WEBCAM)  :
            print("SET TO START °°")
            if self.stream_reader.stop_reading_from_user_action :
                self.stream_reader.stop_reading_from_user_action=False
                print("SET TO START")
                return jsonify(result='stream started')


    def go_to_next_frame(self):
        print("GET NEXT FRAME START °°")
        if self.stream_reader.stop_reading_from_user_action :
            self.stream_reader.get_one_next_frame=True 
            self.stream_reader.stop_reading_from_user_action=False
            print("SET TO NEXT FRAME START")
            return jsonify(result='stream started  +1')
        return jsonify(result='error getting  NEXT FRAME ')


    def start_offline_detection(self,selected_video):
        # wait for streamer to be created before starting
        print(" START OfflineDetection")
        selected_video="videos/"+selected_video
        if (selected_video!= self.stream_reader.video_src):
            self.video_src=selected_video
            self.stream_reader.change_video_file(selected_video)
        self.offline_detector=OfflineDetector(self.detection_service,stream_source=self.stream_source ,video_src=self.video_src ) 
        self.offline_detector.threshold= self.stream_reader.threshold  
        self.offline_detector.nms_threshold=self.stream_reader.nms_threshold
        self.offline_detector.start()
        return jsonify(result='OfflineDetector started')

    def start_offline_tracking(self,selected_video):
        # wait for streamer to be created before starting
        print(" START OfflineTracker")
        selected_video="videos/"+selected_video
        if (selected_video!= self.stream_reader.video_src):
            self.video_src=selected_video
            self.stream_reader.change_video_file(selected_video)
            if self.tracking_service:
                self.tracking_service.reset()
        self.offline_tracker=OfflineTracker(tracking_service=self.tracking_service,stream_source=self.stream_source ,video_src=self.video_src ) 
        self.offline_tracker.threshold= self.stream_reader.threshold  
        self.offline_tracker.nms_threshold=self.stream_reader.nms_threshold
        self.offline_tracker.network_input_size=self.detection_service.network_input_size
        self.offline_tracker.start()
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

    def update_cnn_detector_param(self,param,value):
        if param=='networkInputSize':
            self.detection_service.network_input_size=int(value)
        elif param=='threshold':
            self.detection_service.threshold=float(value)
        elif param=='nmsThreshold':
            self.detection_service.nms_threshold=float(value)

        # else :
        #     if self.stream_source== StreamSourceEnum.RASPBERRY_CAM:
        #         if param=='threshold':
        #             self.raspberry_camera.threshold=float(value)
        #         if param=='nmsThreshold':
        #             self.raspberry_camera.nms_threshold=float(value)
        #     else:
        #         if param=='threshold':
        #             self.stream_reader.threshold=float(value)
        #         if param=='nmsThreshold':
        #             self.stream_reader.nms_threshold=float(value)
                
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

        if self.stream_source== StreamSourceEnum.RASPBERRY_CAM:
            print("=======> main_raspberry_camera_stream")        
            return Response(self.raspberry_camera.read_camera_stream(),mimetype='text/event-stream')

        else:
            print("=======> main_video_stream")
            self.stream_reader=StreamReader(detection_service=self.detection_service, stream_source=self.stream_source ,video_src=self.video_src,save_detectors_results=self.save_detectors_results)        
            self.stream_reader.background_subtractor_service=self.background_subtractor_service
            self.stream_reader.tracking_service=self.tracking_service
            self.stream_reader.hybrid_tracking_service=self.hybrid_tracking_service

            if self.stream_reader.buffer :
                self.stream_reader.startBuffering()
            return Response(self.return_stream(),mimetype='text/event-stream')
 

    def switch_client_stream(self, stream):
        if self.stream_reader!=None:
            if stream == 'CNN_DETECTOR':
                self.stream_reader.current_selected_stream= ClientStreamTypeEnum.CNN_DETECTOR
            elif  stream == 'BACKGROUND_SUBTRACTION':
                self.stream_reader.current_selected_stream= ClientStreamTypeEnum.BACKGROUND_SUBTRACTION
            elif stream == 'TRACKING_STREAM':
                self.stream_reader.current_selected_stream= ClientStreamTypeEnum.TRACKING_STREAM
            elif stream == 'HYBRID_TRACKING_STREAM':
                self.stream_reader.current_selected_stream= ClientStreamTypeEnum.HYBRID_TRACKING_STREAM
        
        elif self.stream_source== StreamSourceEnum.RASPBERRY_CAM:
            if self.raspberry_camera!=None:
                if stream == 'CNN_DETECTOR':
                    self.raspberry_camera.current_selected_stream= ClientStreamTypeEnum.CNN_DETECTOR
                elif  stream == 'BACKGROUND_SUBTRACTION':
                    self.raspberry_camera.current_selected_stream= ClientStreamTypeEnum.BACKGROUND_SUBTRACTION
                elif stream == 'TRACKING_STREAM':
                    self.raspberry_camera.current_selected_stream= ClientStreamTypeEnum.TRACKING_STREAM

        return jsonify(result=stream)

    def track_with(self, param):
        self.tracking_service.reset()
        if param=='background_subtraction':
            self.tracking_service.track_object_by_background_sub=True
            self.tracking_service.track_object_by_cnn_detection=False
        if param=='cnn_detection':
            self.tracking_service.track_object_by_background_sub=False
            self.tracking_service.track_object_by_cnn_detection=True      
        return jsonify(result=param)

    def show_missing_tracks(self, value):
        if value=='true':
            self.tracking_service.show_missing_tracks=True
        else:
            self.tracking_service.show_missing_tracks=False
        return jsonify(result=value)

    def activate_stream_simulation(self, value):
        if value=='true':
            self.stream_reader.activate_stream_simulation=True
        else:
            self.stream_reader.activate_stream_simulation=False
        return jsonify(result=value)
 
    def use_cnn_feature_extraction_on_tracking(self,value):
        if value=='true':
            self.tracking_service.use_cnn_feature_extraction=True
        else:
            self.tracking_service.use_cnn_feature_extraction=False
        self.tracking_service.reset()
        return jsonify(result=value)
        
    def activate_detection_for_tracking(self,value):
        if value=='true':
            self.tracking_service.activate_detection_for_tracking=True
        else:
            self.tracking_service.activate_detection_for_tracking=False
        return jsonify(result=value)
 
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

    
    # def set_video_starting_second(self,second):
    #     # self.stream_reader.set_starting_second(second)
    #     # buffer.video_start_seconde=int(second)
    #     return jsonify(result=second)
