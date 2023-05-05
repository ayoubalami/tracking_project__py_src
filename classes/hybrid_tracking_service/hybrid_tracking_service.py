
import time
import cv2
import numpy as np
from classes.detection_services.detection_service import IDetectionService
from utils_lib.enums import SurveillanceRegionEnum
from classes.background_subtractor_service import BackgroundSubtractorService

class HybridTrackingService():
    
    background_subtractor_service:BackgroundSubtractorService=None
    detection_service:IDetectionService=None   
    d_start,d_height,tr_start,tr_height= 200,100,300,500
    x_start_ratio,x_end_ratio=0,.55
    # d_x_start,d_width,tr_x_start,tr_width=
    is_region_initialization_done=False
    video_resolution_ratio=1
    objects_in_detection_region=[]
    objects_in_tracking_region=[]
    
    debug_surveillance_section=[]
    debug_surveillance_section_right_marge=120

    def __init__(self, detection_service:IDetectionService,background_subtractor_service:BackgroundSubtractorService  ):
        self.background_subtractor_service=background_subtractor_service
        self.detection_service=detection_service
        pass
    
    def apply(self,frame,threshold=0.5,nms_threshold=0.5): 

        start_time=time.perf_counter()

        if self.is_region_initialization_done==False:
            width,height=frame.shape[1],frame.shape[0]
            self.init_regions(width,height )  

        resized_frame  ,raw_detection_data=self.background_subtractor_service.apply(frame=frame,boxes_plotting=False)
        original_frame=resized_frame.copy()
        detection_region , tracking_region=self.divide_draw_frame(resized_frame,raw_detection_data)
        detection_region=self.process_detection_region(detection_region)
        tracking_region=self.process_tracking_region(tracking_region)

        resized_frame=self.consolidate_frame(original_frame,resized_frame,detection_region,tracking_region)

        detection_time=round(time.perf_counter()-start_time,3)
        if round(time.perf_counter()-start_time,3)>0:
            detection_fps=1/round(time.perf_counter()-start_time,3)
        else :
            detection_fps=0
        self.addTrackingAndDetectionTimeAndFPS(resized_frame,detection_time,detection_fps)

        # return foreground_detection_frame, 0 #inference_time
        return resized_frame #inference_time

    #  def getRawDetections(self,origin_frame,threshold=0.5 ,nms_threshold=0.5): 
    #     if self.activate_detection_for_tracking:
    #         if self.track_object_by_cnn_detection==True and self.detection_service !=None and self.detection_service.get_selected_model() !=None:
    #             return  self.detection_service.detect_objects(origin_frame,threshold= threshold ,nms_threshold=nms_threshold,boxes_plotting=False)
    #         if self.track_object_by_background_sub==True and self.background_subtractor_service !=None :
    #             return  self.background_subtractor_service.apply(origin_frame,boxes_plotting=False)
    #     return origin_frame,[]

    # def apply(self,frame): 
    #     if self.is_region_initialization_done==False:
    #         width,height=frame.shape[1],frame.shape[0]
    #         self.init_regions(width,height, self.d_start,self.d_height,self.tr_start,self.tr_height)  
    #     foreground_detection_frame,raw_mask_frame,inference_time=self.background_subtractor_service.apply(frame)
    #     detection_region , tracking_region=self.divide_frame(foreground_detection_frame)

    #     detection_region=self.process_detection_region(detection_region)
    #     tracking_region=self.process_tracking_region(tracking_region)

    #     self.consolidate_frame(foreground_detection_frame,detection_region,tracking_region)
    #     return foreground_detection_frame, 0 #inference_time

    def divide_draw_frame(self,frame,raw_detection_data):
        detection_region= frame[self.d_region_y:self.d_region_y+self.d_region_h, self.d_region_x:self.d_region_x+self.d_region_w]
        tracking_region=  frame[self.tr_region_y:self.tr_region_y+self.tr_region_h, self.tr_region_x:self.tr_region_x+self.tr_region_w]
        # print(raw_detection_data)
        # print('==================')
        for dd in raw_detection_data:
            (x, y, w, h) = dd[0]
            center_x=int(x+w/2)
            center_y=int(y+h/2)

            if center_y>self.d_region_y and center_y<self.d_region_y+self.d_region_h \
            and center_x>self.d_region_x and center_x<self.d_region_x+self.d_region_w : 

                self.objects_in_detection_region.append((x, y, w, h))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), int(2*self.video_resolution_ratio))
                cv2.circle(frame, (center_x,center_y), radius=2, color=(0, 255, 255), thickness=-1)
        
            if center_y>self.tr_region_y and center_y<self.tr_region_y+self.tr_region_h\
            and center_x>self.tr_region_x and center_x<self.tr_region_x+self.tr_region_w:

                self.objects_in_tracking_region.append((x, y, w, h))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), int(2*self.video_resolution_ratio))
                cv2.circle(frame, (center_x,center_y), radius=2, color=(255, 0, 255), thickness=-1)
            
        # print(self.objects_in_detection_region)
        # print(self.objects_in_tracking_region)
        # print("====")
        # self.objects_in_detection_region=[]
        # self.objects_in_tracking_region=[]
        return detection_region , tracking_region

    def consolidate_frame(self,original_frame,frame,detection_region,tracking_region):
        frame[self.d_region_y:self.d_region_y+self.d_region_h, self.d_region_x:self.d_region_x+self.d_region_w]=detection_region
        frame[self.tr_region_y:self.tr_region_y+self.tr_region_h, self.tr_region_x:self.tr_region_x+self.tr_region_w]=tracking_region
       
        frame=self.add_debug_surveillance_section(original_frame,frame)

        
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

    def init_regions(self,width,height ): 
        x_end=int(self.x_end_ratio*width)
        x_start=int(self.x_start_ratio*width)
        (self.d_region_x, self.d_region_y, self.d_region_w, self.d_region_h)     =  x_start,self.d_start,x_end,self.d_height
        (self.tr_region_x, self.tr_region_y, self.tr_region_w, self.tr_region_h) =   x_start,self.tr_start,x_end,self.tr_height
        self.is_region_initialization_done=True

    def add_debug_surveillance_section(self,original_frame,frame):
        detection_section = np.zeros((self.debug_surveillance_section_right_marge, frame.shape[1], 3), dtype=np.uint8)
        tracking_section = np.zeros((self.debug_surveillance_section_right_marge, frame.shape[1], 3), dtype=np.uint8)
        cnn_section = np.zeros((self.debug_surveillance_section_right_marge , frame.shape[1], 3), dtype=np.uint8)

        detection_section[:,:,:] = (224, 223, 245)
        tracking_section[:,:,:] = (245, 227, 213)
        cnn_section[:,:,:] = (145, 227, 213)

        self.add_detected_objects_to_debug_section(original_frame,detection_section)
        self.add_tracked_objects_to_debug_section(original_frame,tracking_section)
        self.add_cnn_objects_to_debug_section(original_frame,cnn_section)
        
        frame = cv2.vconcat([frame, detection_section])
        frame = cv2.vconcat([frame, tracking_section])
        frame = cv2.vconcat([frame, cnn_section])
        return frame

    def add_detected_objects_to_debug_section(self,original_frame,detection_section):
        offset=0
        for roi in self.objects_in_detection_region:
            (x, y, w, h) = roi 
            object = original_frame[y:y+h, x:x+w]
            # cnn_detections=self.detect_ROI_with_CNN(object)
            # if cnn_detections:
            #     self.draw_CNN_bbox_over_ROI(object,cnn_detections)
 
            if offset+w<detection_section.shape[1]:
                detection_section[:h,offset:w+offset] = object[:self.debug_surveillance_section_right_marge,:]
            offset+=w
        # self.objects_in_detection_region=[]

    def add_tracked_objects_to_debug_section(self,original_frame,tracking_section):
        offset=0
        for roi in self.objects_in_tracking_region:
            (x, y, w, h) = roi
            object = original_frame[y:y+h, x:x+w]
            if offset+w<tracking_section.shape[1]:
                tracking_section[:h,offset:w+offset] = object[:self.debug_surveillance_section_right_marge,:]
            offset+=w
        self.objects_in_tracking_region=[]

    def add_cnn_objects_to_debug_section(self,original_frame,cnn_section):
        offset=0
        for roi in self.objects_in_detection_region:
            (x, y, w, h) = roi 
            object = original_frame[y:y+h, x:x+w]
            cnn_detections=self.detect_ROI_with_CNN(object)
            if cnn_detections:
                self.draw_CNN_bbox_over_ROI(object,cnn_detections)

            # print("cnn_detection===")
            # print(cnn_detection)
            if offset+w<cnn_section.shape[1]:
                cnn_section[:h,offset:w+offset] = object[:self.debug_surveillance_section_right_marge,:]
            offset+=w
        self.objects_in_detection_region=[]

    def detect_ROI_with_CNN(self,object,threshold=0.5 ,nms_threshold=0.5): 
        if self.detection_service !=None and self.detection_service.get_selected_model() !=None:
            return  self.detection_service.detect_objects(object,threshold= threshold ,nms_threshold=nms_threshold,boxes_plotting=False)[1]
        return None
    
    def draw_CNN_bbox_over_ROI(self,object,cnn_detections):
        for bbox,confidence,label in cnn_detections:
            x,y,w,h=bbox
            cv2.rectangle(object, (x, y), (x + w, y + h), (25, 55, 155), int(2*self.video_resolution_ratio))
            displayText = '{}: {:.2f}'.format(label, confidence) 
            cv2.putText(object, displayText, ((x + w )//2, (y + h)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25,25,250), 2)

    def addTrackingAndDetectionTimeAndFPS(self,img,detection_time,detection_fps):
        width=img.shape[1]
        cv2.rectangle(img,(int(width-195),10),(int(width-10),85),color=(240,240,240),thickness=-1)
        cv2.putText(img, f'FPS: {round(detection_fps,2)}', (int(width-190),30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (25,25,250), 2)
        cv2.putText(img, f'Det. time: {round(detection_time*1000)}ms', (int(width-190),52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (250,25,25), 2)
        # cv2.putText(img, f'Tra. time: {round(tracking_time*1000)}ms', (int(width-190),73), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (250,25,25), 2)
 