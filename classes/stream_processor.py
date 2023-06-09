

# from app_service  import AppService
import cv2,time,os, numpy as np
import pyshine as ps
from classes.detection_services.detection_service import IDetectionService
from  classes.webcam_reader import WebcamReader
from  classes.buffer import Buffer
from classes.background_subtractor_service import BackgroundSubtractorService
from classes.tracking_service.tracking_service import TrackingService
from classes.hybrid_tracking_service.hybrid_tracking_service import HybridTrackingService
from classes.video_stream import VideoStream
from utils_lib.enums import  StreamSourceEnum
import json
import base64

class StreamProcessor: 

    np.random.seed(123)
    video_stream:VideoStream
    stream_source : StreamSourceEnum
    app_service:None

    def __init__(self,app_service):
        self.stream_source=StreamSourceEnum.FILE
        self.video_stream=VideoStream()
        self.app_service=app_service

        
    def return_stream(self):
        result={}
        for frame in self.video_stream.get_frames():
            if not self.video_stream.stopped or self.video_stream.one_next_frame:
                frame,inference_time= self.process(frame)
                result['detectorStream']=self.encodeStreamingFrame(frame=frame,resize_ratio=1,jpeg_quality=80)
            yield 'event: message\ndata: ' + json.dumps(result) + '\n\n'
            
    def process(self,frame):
        if self.app_service.detection_service !=None  :
            if self.app_service.detection_service.get_selected_model() !=None:
                detection_frame ,inference_time = self.app_service.detection_service.detect_objects(frame)
                return detection_frame,inference_time
        return frame,-1

    def encodeStreamingFrame(self,frame,resize_ratio=1,jpeg_quality=80):
        resized_frame=frame
        if resize_ratio!=1:
            img_width, img_height = self.video_stream.width,self.video_stream.height
            resized_frame=cv2.resize(frame, (int(img_width*resize_ratio) ,int(img_height*resize_ratio) ))
        self.add_frame_time(frame)
        if jpeg_quality!=100:
            ret,buffer=cv2.imencode('.jpg',resized_frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        else:
            ret,buffer=cv2.imencode('.jpg',resized_frame)
        img_bytes=buffer.tobytes()
        return  base64.b64encode(img_bytes).decode()

    def add_frame_time(self,frame):
        if self.stream_source==StreamSourceEnum.FILE :
            minute,second=((int)(self.video_stream.current_video_time))//60,((int)(self.video_stream.current_video_time))%60
            time=str("{:02d}".format(minute))+":"+str( "{:02d}".format(second))
            cv2.putText(frame,time, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255,255,250), 2, cv2.LINE_AA)
        