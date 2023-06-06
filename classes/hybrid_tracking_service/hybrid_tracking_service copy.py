
import time
import cv2
import numpy as np
from classes.detection_services.detection_service import IDetectionService
from utils_lib.enums import SurveillanceRegionEnum
from classes.background_subtractor_service import BackgroundSubtractorService
from skimage.metrics import structural_similarity
import math

class HybridTrackingService():
    
    background_subtractor_service:BackgroundSubtractorService=None
    detection_service:IDetectionService=None   
    # d_start,d_height,tr_start,tr_height= 200,100,300,500

    detection_y_start_ratio,detection_y_end_ratio=0.33,.5
    tracking_y_start_ratio,tracking_y_end_ratio=0.5,1

    detection_x_start_ratio,detection_x_end_ratio=0.33,.52
    tracking_x_start_ratio,tracking_x_end_ratio=0.05,0.52

    is_region_initialization_done=False
    video_resolution_ratio=1
    objects_in_detection_region=[]
    objects_in_tracking_region=[]
    DVR=[]
    debug_surveillance_section=[]
    debug_surveillance_section_right_marge=170
    unique_vehicle_index=0
    DVR_temp=[]
    debug_mode=True

    # orb = cv2.ORB_create()
    orb = cv2.ORB_create(scaleFactor=1.2)
    sift = cv2.SIFT_create() 
    sift_bf = cv2.BFMatcher()
    frame_size=None

    def __init__(self, detection_service:IDetectionService,background_subtractor_service:BackgroundSubtractorService  ):
        self.background_subtractor_service=background_subtractor_service
        # self.background_subtractor_service.video_resolution_ratio=.2

        self.detection_service=detection_service
        pass
    
    def apply(self,frame,threshold=0.5,nms_threshold=0.5): 
        start_time=time.perf_counter()
        if self.is_region_initialization_done==False:
            self.frame_size=frame.shape
            self.init_regions( ) 

        resized_frame  ,raw_detection_data=self.background_subtractor_service.apply(frame=frame,boxes_plotting=False)
        original_frame=resized_frame.copy()
        detection_region , tracking_region=self.divide_draw_frame(resized_frame)

        self.DVR_temp=self.DVR.copy()
        self.process_detection_region(original_frame,resized_frame,raw_detection_data,detection_region)
        self.process_tracking_region(original_frame,resized_frame,raw_detection_data,tracking_region)
        self.DVR=self.DVR_temp

        if self.debug_mode:
            resized_frame=self.add_debug_surveillance_section(original_frame,resized_frame)

        # self.DVR=[]

        detection_time=round(time.perf_counter()-start_time,3)
        if round(time.perf_counter()-start_time,3)>0:
            detection_fps=1/round(time.perf_counter()-start_time,3)
        else :
            detection_fps=0
        self.addTrackingAndDetectionTimeAndFPS(resized_frame,detection_time,detection_fps)

        # return foreground_detection_frame, 0 #inference_time
        return resized_frame #inference_time
 
    def divide_draw_frame(self,frame):
        detection_region= frame[self.detection_y_start_position:self.detection_y_end_position, self.detection_x_start_position:self.detection_x_end_position]
        tracking_region=  frame[self.tracking_y_start_position:self.tracking_y_end_position , self.tracking_x_start_position:self.tracking_x_end_position]
        return detection_region , tracking_region

    def process_detection_region(self,original_frame,frame,raw_detection_data,detection_region):        
        self.objects_in_detection_region=[]
        count_in_detection_region=0
        for dd in raw_detection_data:
            (x, y, w, h) = dd[0]
            center_x=int(x+w/2)
            center_y=int(y+h/2)
            if center_y>self.detection_y_start_position and center_y<self.detection_y_end_position\
            and center_x>self.detection_x_start_position and center_x<self.detection_x_end_position:
                count_in_detection_region+=1

        if count_in_detection_region>0:
            bb_blob = original_frame[self.detection_y_start_position  :self.detection_y_end_position, self.detection_x_start_position:self.detection_x_end_position]
            cnn_detections=self.detect_ROI_with_CNN(bb_blob)
            for cnn_bbox,confidence,label in cnn_detections:
                x,y,w,h=cnn_bbox
                if y>1 and y< self.detection_y_end_position-self.detection_y_start_position-h-2 and x>1 :  
                    absolute_cnn_center_x,absolute_cnn_center_y=self.detection_x_start_position +x+(w//2),self.detection_y_start_position +y+(h//2)
                    absolute_cnn_bbox_x,absolute_cnn_bbox_y =x+self.detection_x_start_position,y+self.detection_y_start_position

                    if absolute_cnn_center_y>self.detection_y_start_position and absolute_cnn_center_y<self.detection_y_end_position\
                    and absolute_cnn_center_x>self.detection_x_start_position and absolute_cnn_center_x<self.detection_x_end_position: 
                        cnn_crop=bb_blob[y:min(self.debug_surveillance_section_right_marge+y,h+y),x:x+w]
                        # cnn_crop
                        border_color=(255,255,0)
                        cv2.rectangle(frame, ( absolute_cnn_bbox_x, absolute_cnn_bbox_y), ( absolute_cnn_bbox_x+w, absolute_cnn_bbox_y+ h), border_color  , 2)
                        id=self.unique_vehicle_index
                        new_vehicle={ 'id':id ,'center_xy':(absolute_cnn_center_x,absolute_cnn_center_y), 'bbox_xywh':(absolute_cnn_bbox_x,absolute_cnn_bbox_y,w ,h) ,'confidence':confidence,'label':label, 'image':cnn_crop,'key_points':[],'description':[], 'region':SurveillanceRegionEnum.DETECTION_REGION  }
                        vehicle_regitred_properties=self.register_to_DVR(new_vehicle,SurveillanceRegionEnum.DETECTION_REGION)
                # vehicle_regitred_properties=self.register_to_DVR(roi=cnn_crop,bbox_xywh=cnn_bbox,confidence=confidence,label=label)

        if self.debug_mode :
            height, width = detection_region.shape[:2]
            colored_rect = np.zeros(detection_region.shape, dtype=np.uint8)
            colored_rect[:, :, 0] = 200
            border_color=(255,0,0)
            colored_detection_region = cv2.addWeighted(detection_region, 1, colored_rect, 0.5, 1.0)
            frame[self.detection_y_start_position :self.detection_y_end_position, self.detection_x_start_position:self.detection_x_end_position] = colored_detection_region
            cv2.rectangle(frame, ( self.detection_x_start_position,self.detection_y_start_position), (self.detection_x_start_position+width, self.detection_y_start_position+height), border_color , 2)
            #  DRAW DETECTION BBOX 
            for (x, y, w, h) in self.objects_in_detection_region:
                center_x=int(x+w/2)
                center_y=int(y+h/2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), int(2*self.video_resolution_ratio))
                cv2.circle(frame, (center_x,center_y), radius=2, color=(0, 255, 255), thickness=-1)

        
    # def process_detection_region(self,original_frame,frame,raw_detection_data,detection_region):        
    #     self.objects_in_detection_region=[]
    #     for dd in raw_detection_data:
    #         (x, y, w, h) = dd[0]
    #         center_x=int(x+w/2)
    #         center_y=int(y+h/2)
    #         if center_y>self.detection_y_start_position and center_y<self.detection_y_end_position\
    #             and center_x>self.x_start_position and center_x<self.x_end_position: 
    #             self.objects_in_detection_region.append((x, y, w, h))
                
    #             # bb_blob = original_frame[y:y+h, x:x+w]
    #             bb_blob = original_frame[y:y+h, x:x+w]
                
    #             cnn_detections=self.detect_ROI_with_CNN(bb_blob)
    #             for cnn_bbox,confidence,label in cnn_detections:
    #                 relative_cnn_bbox_x,relative_cnn_bbox_y,cnn_bbox_w,cnn_bbox_h=cnn_bbox
    #                 absolute_cnn_center_x,absolute_cnn_center_y=x+relative_cnn_bbox_x+(cnn_bbox_w)//2,y+relative_cnn_bbox_y+(cnn_bbox_h)//2
    #                 absolute_cnn_bbox_x,absolute_cnn_bbox_y =relative_cnn_bbox_x+x,relative_cnn_bbox_y+y
                    
    #                 if absolute_cnn_center_y>self.detection_y_start_position and absolute_cnn_center_y<self.detection_y_end_position\
    #                 and absolute_cnn_center_x>self.x_start_position and absolute_cnn_center_x<self.x_end_position: 
    #                     cnn_crop=bb_blob[relative_cnn_bbox_y:min(self.debug_surveillance_section_right_marge+relative_cnn_bbox_y,cnn_bbox_h+relative_cnn_bbox_y),relative_cnn_bbox_x:relative_cnn_bbox_x+cnn_bbox_w]
    #                     id=self.unique_vehicle_index
    #                     new_vehicle={ 'id':id ,'center_xy':(absolute_cnn_center_x,absolute_cnn_center_y), 'bbox_xywh':(absolute_cnn_bbox_x,absolute_cnn_bbox_y,cnn_bbox_w ,cnn_bbox_h) ,'confidence':confidence,'label':label, 'image':cnn_crop,'key_points':[],'description':[], 'region':SurveillanceRegionEnum.DETECTION_REGION  }
    #                     vehicle_regitred_properties=self.register_to_DVR(new_vehicle,SurveillanceRegionEnum.DETECTION_REGION)
    #                     # vehicle_regitred_properties=self.register_to_DVR(roi=cnn_crop,bbox_xywh=cnn_bbox,confidence=confidence,label=label)

    #     if self.debug_mode :
    #         height, width = detection_region.shape[:2]
    #         colored_rect = np.zeros(detection_region.shape, dtype=np.uint8)
    #         colored_rect[:, :, 0] = 200
    #         border_color=(255,0,0)
    #         colored_detection_region = cv2.addWeighted(detection_region, 1, colored_rect, 0.5, 1.0)
    #         frame[self.detection_y_start_position :self.detection_y_end_position, self.x_start_position:self.x_end_position] = colored_detection_region
    #         cv2.rectangle(frame, ( self.x_start_position,self.detection_y_start_position), (self.x_start_position+width, self.detection_y_start_position+height), border_color , 2)
    #         #  DRAW DETECTION BBOX 
    #         for (x, y, w, h) in self.objects_in_detection_region:
    #             center_x=int(x+w/2)
    #             center_y=int(y+h/2)
    #             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), int(2*self.video_resolution_ratio))
    #             cv2.circle(frame, (center_x,center_y), radius=2, color=(0, 255, 255), thickness=-1)


    def process_tracking_region(self,original_frame,frame,raw_detection_data,tracking_region ):     
        self.objects_in_tracking_region=[]
        for dd in raw_detection_data:
            (x, y, w, h) = dd[0]
            center_x=int(x+w/2)
            center_y=int(y+h/2)
            if center_y>self.tracking_y_start_position and center_y<self.tracking_y_end_position \
            and center_x>self.tracking_x_start_position and center_x<self.tracking_x_end_position :
                self.objects_in_tracking_region.append((x, y, w, h))
                bb_blob = original_frame[y:y+h, x:x+w]

                id=self.unique_vehicle_index
                new_vehicle={ 'id':id ,'center_xy':(center_x,center_y), 'bbox_xywh':(x, y, w, h) ,'confidence':-1,'label':None, 'image':bb_blob,'key_points':[],'description':[], 'region':SurveillanceRegionEnum.TRACKING_REGION  }
                vehicle_regitred_properties=self.register_to_DVR(new_vehicle,SurveillanceRegionEnum.TRACKING_REGION)
    
        if self.debug_mode:
            height, width = tracking_region.shape[:2]
            colored_rect = np.zeros(tracking_region.shape, dtype=np.uint8) 
            colored_rect[:, :, 2] = 200
            border_color=(0,0,255)
            colored_tracking_region= cv2.addWeighted(tracking_region, 1, colored_rect, 0.5, 1.0)
            frame[self.tracking_y_start_position :self.tracking_y_end_position, self.tracking_x_start_position:self.tracking_x_end_position] = colored_tracking_region
            cv2.rectangle(frame, ( self.tracking_x_start_position,self.tracking_y_start_position), (self.tracking_x_start_position+width, self.tracking_y_start_position+height), border_color  , 2)
            #  DRAW TRACKING BBOX 
        
        for (x, y, w, h)  in self.objects_in_tracking_region:
            center_x=int(x+w/2)
            center_y=int(y+h/2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), int(2*self.video_resolution_ratio))
            cv2.circle(frame, (center_x,center_y), radius=2, color=(255, 0, 255), thickness=-1)

    def init_regions(self ): 
        width,height=self.frame_size[1],self.frame_size[0]
        # self.x_start_position=int(self.x_start_ratio*width)
        # self.x_end_position=int(self.x_end_ratio*width)

        self.detection_x_start_position=int(self.detection_x_start_ratio*width)
        self.detection_x_end_position=int(self.detection_x_end_ratio*width)
        self.tracking_x_start_position=int(self.tracking_x_start_ratio*width)
        self.tracking_x_end_position=int(self.tracking_x_end_ratio*width)

        self.detection_y_start_position=int(self.detection_y_start_ratio*height)
        self.detection_y_end_position=int(self.detection_y_end_ratio*height)
        self.tracking_y_start_position=int(self.tracking_y_start_ratio*height)
        self.tracking_y_end_position=int(self.tracking_y_end_ratio*height)


        self.is_region_initialization_done=True

    def add_debug_surveillance_section(self,original_frame,frame):
        detection_section = np.zeros((self.debug_surveillance_section_right_marge, frame.shape[1], 3), dtype=np.uint8)
        tracking_section = np.zeros((self.debug_surveillance_section_right_marge, frame.shape[1], 3), dtype=np.uint8)
        cnn_section = np.zeros((self.debug_surveillance_section_right_marge , frame.shape[1], 3), dtype=np.uint8)

        detection_section[:,:,:] = (245, 227, 213)
        tracking_section[:,:,:] = (224, 223, 245)
        cnn_section[:,:,:] = (145, 227, 213)

        self.add_detected_objects_to_debug_section(original_frame,detection_section)
        self.add_tracked_objects_to_debug_section(original_frame,tracking_section)
        self.add_cnn_objects_to_debug_section(cnn_section)
        
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

    def add_tracked_objects_to_debug_section(self,original_frame,tracking_section):
        offset=0
        for roi in self.objects_in_tracking_region:
            (x, y, w, h) = roi
            object = original_frame[y:y+h, x:x+w]
            if offset+w<tracking_section.shape[1]:
                tracking_section[:h,offset:w+offset] = object[:self.debug_surveillance_section_right_marge,:]
            offset+=w

    def add_cnn_objects_to_debug_section(self,cnn_section):
        offset=0
        for cnn_object in self.DVR:
            (bbox_x,bbox_y,bbox_w,bbox_h)=cnn_object['bbox_xywh']

            if bbox_h>self.debug_surveillance_section_right_marge-90:
                resized_crop=cv2.resize(cnn_object['image'], (self.debug_surveillance_section_right_marge-90, self.debug_surveillance_section_right_marge-90))
                bbox_h,bbox_w=resized_crop.shape[:2]
            else:
                resized_crop=cnn_object['image']

            if offset+resized_crop.shape[1]<cnn_section.shape[1]:
                region='Detection Region' if cnn_object['region']==SurveillanceRegionEnum.DETECTION_REGION else 'Tracking region'
                # cnn_section[:cnn_object['image'].shape[0],offset:cnn_object['image'].shape[1]+offset]=cnn_object['image'][:self.debug_surveillance_section_right_marge,:]         
                cnn_section[:resized_crop.shape[0],offset:resized_crop.shape[1]+offset]=resized_crop[:self.debug_surveillance_section_right_marge,:]
                # [cnn_bbox_y:min(self.debug_surveillance_section_right_marge+cnn_bbox_y,cnn_bbox_h+cnn_bbox_y),cnn_bbox_x:cnn_bbox_x+cnn_bbox_w]
                cv2.putText(cnn_section, cnn_object['label'], ((offset ) , ( bbox_h) + 12 ), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25,25,250), 1)
                cv2.putText(cnn_section,  str(round(cnn_object['confidence'],2)), ((offset ) , ( bbox_h) + 28 ), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25,25,250), 1)
                # cv2.putText(cnn_section, str(len(cnn_object['key_points'])), ((offset ) , ( bbox_h) + 44 ), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (250,25,50), 2)
                cv2.putText(cnn_section,  region, ((offset ) , ( bbox_h) + 44 ), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (250,25,50), 2)
                cv2.putText(cnn_section, "ID : "+str(cnn_object['id']), ((offset ) , ( bbox_h) + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (250,125,50), 2)
                cv2.putText(cnn_section, str(cnn_object['status']), ((offset ) , ( bbox_h) + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (250,125,150), 2)
                
                offset+=bbox_w+35

    def detect_ROI_with_CNN(self,object): 
        if self.detection_service !=None and self.detection_service.get_selected_model() !=None:
            return  self.detection_service.detect_objects(object,boxes_plotting=False)[1]
        return []
    
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


    def update_DVR_properties(self,new_vehicle,registerd_vehicle,region):
        # new_vehicle['id']=registerd_vehicle['id']
        vehicle_to_update= next(filter(lambda vehicle: vehicle['id'] == registerd_vehicle['id'], self.DVR_temp))
        self.DVR.remove(registerd_vehicle)
        vehicle_to_update['bbox_xywh']=new_vehicle['bbox_xywh']

        if region==SurveillanceRegionEnum.DETECTION_REGION:
            vehicle_to_update['confidence']=new_vehicle['confidence']
            vehicle_to_update['label']=new_vehicle['label']
    
        vehicle_to_update['image']=new_vehicle['image']
        vehicle_to_update['key_points']=new_vehicle['key_points']
        vehicle_to_update['description']=new_vehicle['description']
        vehicle_to_update['center_xy']=new_vehicle['center_xy']
        vehicle_to_update['status']=new_vehicle['status']

        x,y,w,h =new_vehicle['bbox_xywh'] 
        # print("math.fabs(y-self.tracking_y_end_position<10)")
        # print(new_vehicle['bbox_xywh'] )
        # # print(y-self.tracking_y_end_position)

        # print(math.fabs(y+h-self.tracking_y_end_position))
        # print(h)
        print(math.fabs(y+h-self.tracking_y_end_position),h)
        if  math.fabs(y+h-self.tracking_y_end_position)<10 and  h <=150:
            vehicle_to_update['status']='OUT'

        # print(math.fabs(cnn_center_y- self.detection_y_end_position))
        # if math.fabs(cnn_center_y- self.detection_y_end_position)<50 : 
        #     vehicle_to_update['region']=SurveillanceRegionEnum.TRACKING_REGION
        #     vehicle_to_update['region']='tracking'

    def register_to_DVR(self,new_vehicle,surveillance_region:SurveillanceRegionEnum):
        # kps, des = self.orb.detectAndCompute(cnn_crop, None)
        # cnn_crop = cv2.drawKeypoints(cnn_crop, kps, None)
        # kps, des = self.sift.detectAndCompute(roi, None)
        # new_vehicle={ 'id':id ,'bbox_xywh':bbox_xywh ,'confidence':confidence,'label':label, 'image':roi,'key_points':kps,'description':des, 'region':'detection'  }


        # FIX THIS 
        # print(new_vehicle['id'] )
        # print(new_vehicle['region'] )
        # print(new_vehicle['bbox_xywh'] )
        # print(new_vehicle['center_xy'] )

        if surveillance_region==SurveillanceRegionEnum.DETECTION_REGION:
            similare_vehicle=self.check_for_similarity_in_detection_region(new_vehicle)
            if similare_vehicle:
                new_vehicle['status']='TRACKED_DET_REG'
                self.update_DVR_properties(new_vehicle,similare_vehicle,SurveillanceRegionEnum.DETECTION_REGION)
            else:
                new_vehicle['status']='DETECTED'
                self.DVR_temp.append(new_vehicle)
                self.unique_vehicle_index=self.unique_vehicle_index+1
            return new_vehicle

        if surveillance_region==SurveillanceRegionEnum.TRACKING_REGION:

            similare_vehicle=self.check_for_similarity_in_tracking_region(new_vehicle)
            if similare_vehicle:
                new_vehicle['status']='TRACKED_TR_REG'
                self.update_DVR_properties(new_vehicle,similare_vehicle,SurveillanceRegionEnum.TRACKING_REGION)
            else:
                new_vehicle['status']='MISSED'

    def check_for_similarity_in_tracking_region(self,current_vehicle):
        current_vehicle_image=current_vehicle['image']
        euclidean_distance_min_score=100000
        euclidean_distance_min_vehicle=None
        for registred_vehicle in self.DVR:

            if registred_vehicle['status']=='OUT':
                continue
            x,y=registred_vehicle['center_xy']
            if registred_vehicle['region']!=SurveillanceRegionEnum.TRACKING_REGION  \
            and y+50<self.tracking_y_start_position:
                continue
            distance_score=self.euclidean_similarity(current_vehicle,registred_vehicle)
            # print(" [TRACKING REGION TEST] DVR VEHICLE ID " + str(current_vehicle['id']) +" <=> "+ str(registred_vehicle['id'] ))
            # print(distance_score)
            if distance_score<euclidean_distance_min_score and distance_score!=-1:
                euclidean_distance_min_score=distance_score
                euclidean_distance_min_vehicle=registred_vehicle

        if (euclidean_distance_min_vehicle!=None ):
            # print('<<<TRACKING REGION  MATCH>>>')
            return euclidean_distance_min_vehicle
        else:
            # print('<<<TRACKING REGION  MATCH NOT FOUND>>>')
            return None

    def check_for_similarity_in_detection_region(self,current_vehicle):

        current_vehicle_image=current_vehicle['image']
        structural_similarity_max_score=0
        structural_similarity_max_vehicle=None
        euclidean_distance_min_score=100000
        euclidean_distance_min_vehicle=None
         
        for registred_vehicle in self.DVR:
            if registred_vehicle['status']=='OUT':
                continue
            # ADD shape similarity (width, height) comparison
            if registred_vehicle['region']!=SurveillanceRegionEnum.DETECTION_REGION :
                continue
            registred_vehicle_image= cv2.resize(registred_vehicle['image'], (current_vehicle_image.shape[1], current_vehicle_image.shape[0]))

            # start_time=time.perf_counter()
            ssim = structural_similarity(current_vehicle_image, registred_vehicle_image, multichannel=True,channel_axis=-1)
            # print(time.perf_counter()-start_time)
            # print("time.perf_counter()-start_time")


            # ssim = 1
            distance_score=self.euclidean_similarity(current_vehicle,registred_vehicle)
            
            # print(" TEST DVR VEHICLE ID "+ str(registred_vehicle['id'] ))
            # print(" [DETECTION REGION TEST] DVR VEHICLE ID " + str(current_vehicle['id']) +" <=> "+ str(registred_vehicle['id'] ))
            # print(ssim)
            # print(distance_score)
            # print(registred_vehicle['center_xy'])
            # print(current_vehicle['center_xy'])
            # print("=====")
            
            if ssim>structural_similarity_max_score :
                structural_similarity_max_score=ssim
                structural_similarity_max_vehicle=registred_vehicle

            if distance_score<euclidean_distance_min_score and distance_score!=-1:
                euclidean_distance_min_score=distance_score
                euclidean_distance_min_vehicle=registred_vehicle

        # print("======== max_similarity for vehicle id = " + str(current_vehicle['id'] ))
        # print(structural_similarity_max_score)
        # print(euclidean_distance_min_score)
        # if structural_similarity_max_vehicle:
        #     print(structural_similarity_max_vehicle['id'])
        # if euclidean_distance_min_vehicle:
        #     print(euclidean_distance_min_vehicle['id'])
        
        if (structural_similarity_max_vehicle!=None and structural_similarity_max_vehicle==euclidean_distance_min_vehicle):
            # print('<<<MATCH>>>')
            return structural_similarity_max_vehicle
        else:
            # print('<<<MATCH NOT FOUND>>>')
            return None
    
    def euclidean_similarity(self,vehicle_1,vehicle_2):
        epsilon=1e-4
        # (x1,y1,w1,h1)= vehicle_1['bbox_xywh']
        # (x2,y2,w2,h2)= vehicle_2['bbox_xywh']
        # center_x_1=int(x1+w1/2);center_y_1=int(y1+h1/2)
        # center_x_2=int(x2+w2/2);center_y_2=int(y2+h2/2)     
        center_x_1,center_y_1=vehicle_1['center_xy']
        center_x_2,center_y_2=vehicle_2['center_xy']
        # score_x=(center_x_1-center_x_2)/self.frame_size[1]
        # score_y=(center_y_1-center_y_2)/self.frame_size[0]

        score_x= math.fabs(center_x_1-center_x_2) 
        score_y= (center_y_1-center_y_2) 

        if score_y+10<0:
            return -1
        return score_x+score_y
        # return score_x,score_y
        # return 1/(math.sqrt((center_x_2 - center_x_1)**2 + (center_y_2 - center_y_1)**2)+epsilon)
        # return math.sqrt((center_x_2 - center_x_1)**2 + (center_y_2 - center_y_1)**2)

    def reset(self):
        self.DVR=[]
        self.DVR_temp=[]
        self.objects_in_detection_region=[]
        self.objects_in_tracking_region=[]
        self.debug_surveillance_section=[]
        self.unique_vehicle_index=0
        
