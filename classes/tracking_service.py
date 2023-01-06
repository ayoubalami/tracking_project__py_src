
import time
import cv2
import numpy as np
from utils_lib.enums import SurveillanceRegionEnum
from classes.background_subtractor_service import BackgroundSubtractorService


class TrackingService():
    
    background_subtractor_service:BackgroundSubtractorService=None    
    (d_region_x, d_region_y, d_region_w, d_region_h)=400 , 200,  160 , 160
    (tr_region_x, tr_region_y, tr_region_w, tr_region_h)=800 , 500,  160 , 160

    def __init__(self,background_subtractor_service:BackgroundSubtractorService):
        self.background_subtractor_service=background_subtractor_service
        pass
    
    def apply(self,frame): 
        
        foreground_detection_frame,raw_mask_frame,inference_time=self.background_subtractor_service.apply(frame)
        detection_region , tracking_region=self.divide_frame(foreground_detection_frame)

        detection_region=self.process_detection_region(detection_region)
        tracking_region=self.process_tracking_region(tracking_region)

        self.consolidate_frame(foreground_detection_frame,detection_region,tracking_region)
        return foreground_detection_frame, 0 #inference_time

    def divide_frame(self,frame):
        detection_region= frame[self.d_region_y:self.d_region_y+self.d_region_h, self.d_region_x:self.d_region_x+self.d_region_w]
        tracking_region=  frame[self.tr_region_y:self.tr_region_y+self.tr_region_h, self.tr_region_x:self.tr_region_x+self.tr_region_w]
        return detection_region , tracking_region

    def consolidate_frame(self,frame,detection_region,tracking_region):
        frame[self.d_region_y:self.d_region_y+self.d_region_h, self.d_region_x:self.d_region_x+self.d_region_w]=detection_region
        frame[self.tr_region_y:self.tr_region_y+self.tr_region_h, self.tr_region_x:self.tr_region_x+self.tr_region_w]=tracking_region
        return frame

    def process_detection_region(self,detection_region):        
        return self.color_region(detection_region,SurveillanceRegionEnum.DETECTION_REGION)
           
    def process_tracking_region(self,tracking_region):        
        return self.color_region(tracking_region,SurveillanceRegionEnum.TRACKING_REGION)
    
    def color_region(self,region,region_type):
        height, width = region.shape[:2]
        colored_rect = np.zeros(region.shape, dtype=np.uint8) 
        border_color=(0,0,0)
        if region_type==SurveillanceRegionEnum.DETECTION_REGION:
            colored_rect[:, :, 0] = 200
            border_color=(200, 200, 200)
        elif region_type==SurveillanceRegionEnum.TRACKING_REGION:
            colored_rect[:, :, 2] = 200
            border_color=(200, 200, 200)
        region = cv2.addWeighted(region, 1, colored_rect, 0.5, 1.0)
        cv2.rectangle(region, (0, 0), (width, height), border_color  , 2)
        return region

    def init_regions(self,width,height, d_start,d_height,tr_start,tr_height):
        (self.d_region_x, self.d_region_y, self.d_region_w, self.d_region_h)     =  0,d_start,width,d_height
        (self.tr_region_x, self.tr_region_y, self.tr_region_w, self.tr_region_h) =   0,tr_start,width,tr_height
         