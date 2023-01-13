
from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder
import json
import time
import cv2
from utils_lib.enums import ClientStreamTypeEnum
import base64

# from utils_lib.utils_functions import encodeStreamingFrame,applyDetection

from classes.background_subtractor_service import BackgroundSubtractorService
from classes.tracking_service import TrackingService
from classes.detection_service import IDetectionService

class RaspberryCameraReader :
    def __init__(self,detection_service:IDetectionService,background_subtractor_service:BackgroundSubtractorService,tracking_service:TrackingService): 
        self.start_reading_action=False
        self.detection_service=detection_service
        self.current_selected_stream: ClientStreamTypeEnum=ClientStreamTypeEnum.CNN_DETECTOR
        self.background_subtractor_service=background_subtractor_service
        self.tracking_service=tracking_service
        self.threshold=0.5
        self.nms_threshold=0.5
        try:
            self.picam2 = Picamera2()
        except:
            print("camera not detected program exited....")
            exit()
        self.picam2.configure(self.picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (1296,972)}))
        time.sleep(.1)

    def read_camera_stream(self):
        print("Start READING FROM raspberry camera.||.....")
        try:
            self.picam2.start()
        except:
            self.picam2.stop()
            time.sleep(.2)
            self.picam2.start()

        while(True):
            if self.start_reading_action==False:
                time.sleep(.1)
                continue
            image = self.picam2.capture_array()
            # detection_frame,inference_time=self.applyDetection(image,self.detection_service,threshold=self.threshold,nms_threshold=self.nms_threshold)

            if not self.detection_service :
                time.sleep(.02)
            yield from self.ProcessAndYieldFrame(image)
      
        # self.picam2.stop()
        # print("Stop READING FROM raspberry camera.||.....")
    
    def ProcessAndYieldFrame(self,frame):
        result={}
        copy_frame=frame.copy()

        if self.current_selected_stream== ClientStreamTypeEnum.CNN_DETECTOR:
            detection_frame,inference_time=self.applyDetection(copy_frame)
            result['detectorStream']=self.encodeStreamingFrame(frame=detection_frame,resize_ratio=1,jpeg_quality=80)

        elif self.current_selected_stream== ClientStreamTypeEnum.BACKGROUND_SUBTRACTION:
            foreground_detection_frame,raw_mask_frame,inference_time=self.background_subtractor_service.apply(copy_frame)
            result['backgroundSubStream_1']=self.encodeStreamingFrame(frame=raw_mask_frame,resize_ratio=1,jpeg_quality=80)
            result['backgroundSubStream_2']=self.encodeStreamingFrame(frame=foreground_detection_frame,resize_ratio=1,jpeg_quality=80)
        
        elif self.current_selected_stream== ClientStreamTypeEnum.TRACKING_STREAM:
            tracking_frame,inference_time=self.tracking_service.apply(copy_frame)
            result['trackingStream_1']=self.encodeStreamingFrame(frame=tracking_frame,resize_ratio=1,jpeg_quality=80)
            # result['trackingStream_2']=self.encodeStreamingFrame(frame=tracking_frame,resize_ratio=1,jpeg_quality=50)
            # result['testStream_first']=self.test_stream(origin_frame.copy(),detection_fps)
        yield 'event: message\ndata: ' + json.dumps(result) + '\n\n'

    def encodeStreamingFrame(self,frame,resize_ratio=1,jpeg_quality=100):
        if resize_ratio!=1:
            img_width, img_height = frame.shape[1], frame.shape[0]
            frame=cv2.resize(frame, (int(img_width*resize_ratio) ,int(img_height*resize_ratio) ))
        ret,buffer=cv2.imencode('.jpg',frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        img_bytes=buffer.tobytes()
        return  base64.b64encode(img_bytes).decode()

    def applyDetection(self,origin_frame):   
        # ret,origin_frame=cv2.imencode('.jpg',origin_frame, [cv2.IMWRITE_JPEG_QUALITY, 30])
        # resize_ratio=.5
        # origin_frame=cv2.resize(origin_frame, (int(self.buffer.width*resize_ratio) ,int(self.buffer.height*resize_ratio) ))
        if self.detection_service !=None  and self.detection_service.get_selected_model() !=None:
            detection_frame ,inference_time = self.detection_service.detect_objects(origin_frame, threshold= self.threshold ,nms_threshold=self.nms_threshold)
            return detection_frame,inference_time
        return origin_frame,-1