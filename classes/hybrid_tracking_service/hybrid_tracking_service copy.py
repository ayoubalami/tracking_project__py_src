
from collections import deque
import colorsys
import random
import time
import cv2
import numpy as np
from classes.detection_services.detection_service import IDetectionService
from utils_lib.enums import SurveillanceRegionEnum
from classes.background_subtractor_service import BackgroundSubtractorService
from skimage.metrics import structural_similarity
import math
from scipy.optimize import linear_sum_assignment

class HybridTrackingService():
    
    background_subtractor_service:BackgroundSubtractorService=None
    detection_service:IDetectionService=None   

    # detection_y_start_ratio,detection_y_end_ratio=0.20,.30
    # tracking_y_start_ratio,tracking_y_end_ratio=0.30,1
    # detection_x_start_ratio,detection_x_end_ratio=0.2,.37
    # tracking_x_start_ratio,tracking_x_end_ratio=0.05,0.37

    detection_y_start_ratio,detection_y_end_ratio=0.30,.40
    tracking_y_start_ratio,tracking_y_end_ratio=0.40,1
    detection_x_start_ratio,detection_x_end_ratio=0.33,.52
    tracking_x_start_ratio,tracking_x_end_ratio=0.05,0.52

    additional_marge_of_large_objects=20

    is_region_initialization_done=False
    video_resolution_ratio=1
    objects_in_detection_region=[]
    DVR=[]
    DVR_vehicle_id=0
    # debug_surveillance_section=[]
    debug_surveillance_section_height=170
    debug_mode=True
    max_timeout=100
    max_timeout_of_missing=7
    x_padding=0
    y_padding=70

    # orb = cv2.ORB_create()
    sift = cv2.SIFT_create() 
    sift_bf = cv2.BFMatcher()
    frame_size=None
    similarity_threshold=0.01

    DVR_remaining_after_pretracked_assignment=[]
    DVR_remaining_after_tracked_assignment=[]
    DVR_remaining_after_detected_assignment=[]
    current_tracked_objects=[]
    current_detected_objects=[]
    max_point=100
    colors=[(255,255,255) for _ in range(1000)]
    cnn_inference_count=0
    # feature_extractor_model_file='utils_lib/deep_sort/feature_extractors/mars-small128.pb'
    # encoder=gdet.create_box_encoder(model_filename=feature_extractor_model_file,batch_size=1)

    INF=100000

    def __init__(self, detection_service:IDetectionService,background_subtractor_service:BackgroundSubtractorService  ):

        self.center_pts= [deque(maxlen=self.max_point) for _ in range(1000)]
        self.background_subtractor_service=background_subtractor_service
        self.detection_service=detection_service
        pass
    
    def apply(self,frame): 
        start_time=time.perf_counter()
        if not self.is_region_initialization_done:
            self.init_regions(frame.shape)
            self.is_region_initialization_done=True

        resized_frame  ,raw_detection_data=self.background_subtractor_service.apply(frame=frame,boxes_plotting=False)
        original_frame=resized_frame.copy()

        # if self.debug_mode:
        detection_region , tracking_region=self.divide_draw_frame(resized_frame)

        #TODO DELETE TRACKED FROM DETECTION

        self.process_tracking_region(original_frame,resized_frame,raw_detection_data,tracking_region)
        self.process_detection_region(original_frame,resized_frame,raw_detection_data,detection_region)

        if self.debug_mode:
            resized_frame=self.add_debug_surveillance_section(original_frame,resized_frame)

        self.remove_exited_vehicles()
        self.delete_missed_vehicles()
        self.draw_tracking_points(resized_frame)

        detection_time=round(time.perf_counter()-start_time,3)
        if round(time.perf_counter()-start_time,3)>0:
            detection_fps=1/round(time.perf_counter()-start_time,3)
        else :
            detection_fps=0
        self.addTrackingAndDetectionTimeAndFPS(resized_frame,detection_time,detection_fps)

        
        # return foreground_detection_frame, 0 #inference_time
        return resized_frame #inference_time
 
    

    def remove_exited_vehicles(self):
        for v in self.DVR:
            if v['status']=='EXITED':
                self.center_pts[v['id']]=deque(maxlen=self.max_point) 
                self.DVR.remove(v)
                
    def delete_missed_vehicles(self):
        for v in self.DVR:
            if v['tracklet_updated']==False:
                v['timeout']+=1
            else:
                v['timeout']=0
            # print(v['timeout'])
            v['tracklet_updated']=False
            if (['status']=='MISSED' and v['missing_count']>self.max_timeout_of_missing )or v['timeout']>self.max_timeout:
                self.center_pts[v['id']]=deque(maxlen=self.max_point) 
                self.DVR.remove(v)

        
    def divide_draw_frame(self,frame):
        detection_region= frame[self.detection_y_start_position:self.detection_y_end_position, self.detection_x_start_position:self.detection_x_end_position]
        tracking_region=  frame[self.tracking_y_start_position:self.tracking_y_end_position , self.tracking_x_start_position:self.tracking_x_end_position]
        return detection_region , tracking_region

    def process_detection_region(self,original_frame,frame,raw_detection_data,detection_region):        
        self.objects_in_detection_region=[]
        self.current_detected_objects=[]
        is_blobs_founded_in_detection_zone=False

        for dd in raw_detection_data:
            (x, y, w, h) = dd[0]
            center_x=int(x+w/2)
            center_y=int(y+h/2)
            if center_y>self.detection_y_start_position and center_y<self.detection_y_end_position\
            and center_x>self.detection_x_start_position and center_x<self.detection_x_end_position:
                is_blobs_founded_in_detection_zone=True
                break

        if is_blobs_founded_in_detection_zone:
            bb_blob = original_frame[self.detection_y_start_position -self.y_padding :self.detection_y_end_position +self.y_padding, self.detection_x_start_position:self.detection_x_end_position]
            cnn_detections=self.detect_ROI_with_CNN(bb_blob)
            y_center_detection_zone=self.detection_y_start_position+((self.detection_y_end_position-self.detection_y_start_position)//2)

            for cnn_bbox,confidence,label in cnn_detections:
                x,y,w,h=cnn_bbox               
                y=y-self.y_padding 
                if y<0:
                    continue

                absolute_cnn_center_x,absolute_cnn_center_y=self.detection_x_start_position +x+(w//2),self.detection_y_start_position +y+(h//2)
                absolute_cnn_bbox_x,absolute_cnn_bbox_y =x+self.detection_x_start_position,y+self.detection_y_start_position

                is_object_in_y_center=math.fabs(y_center_detection_zone-absolute_cnn_center_y)<25
                is_a_large_vehicle= h-(self.detection_y_end_position-self.detection_y_start_position)>-10
                more_marge_for_large_object=self.additional_marge_of_large_objects if is_object_in_y_center and is_a_large_vehicle else 0

                # print("&&&&&&&&&&&&&&&" )
                # print(is_object_in_y_center )
                # print(is_a_large_vehicle )
                # print(more_marge_for_large_object)
                # print(y+h)
                # print(self.detection_y_end_position-self.detection_y_start_position +more_marge_for_large_object)
                # print("&&&&&&&&&&&&&&&" )

                # print(y+h)
                # print(self.detection_y_end_position-self.detection_y_start_position +more_marge_for_large_object)
                if y>1  and ( y+h< self.detection_y_end_position-self.detection_y_start_position +more_marge_for_large_object-1 )  and x>1 :  
                    cnn_crop=bb_blob[y+self.y_padding:min(self.debug_surveillance_section_height+y+self.y_padding,h+y+self.y_padding),x:x+w]
                    border_color=(255,255,0)
                    cv2.rectangle(frame, ( absolute_cnn_bbox_x, absolute_cnn_bbox_y), ( absolute_cnn_bbox_x+w, absolute_cnn_bbox_y+ h), border_color  , 2)
                    kps, des = self.sift.detectAndCompute(cnn_crop, None)
                    new_vehicle={ 'id':-1 ,'center_xy':(absolute_cnn_center_x,absolute_cnn_center_y), 'bbox_xywh':(absolute_cnn_bbox_x,absolute_cnn_bbox_y,w ,h) ,'confidence':confidence,'label':label, 'image':cnn_crop,'key_points':kps,'description':des, 'region':SurveillanceRegionEnum.DETECTION_REGION ,'status':'DETECTED' ,'missing_count':0,'speed':0,'is_a_large_vehicle':is_a_large_vehicle,'tracklet_updated':True,'timeout':0}
                    self.current_detected_objects.append(new_vehicle)

        self.assigne_pre_tracked_objects()
        self.assigne_detected_objects()

        if self.debug_mode :
            height, width = detection_region.shape[:2]
            colored_rect = np.zeros(detection_region.shape, dtype=np.uint8)
            colored_rect[:, :, 0] = 200
            border_color=(255,0,0)
            colored_detection_region = cv2.addWeighted(detection_region, 1, colored_rect, 0.5, 1.0)
            frame[self.detection_y_start_position :self.detection_y_end_position, self.detection_x_start_position:self.detection_x_end_position] = colored_detection_region
            cv2.rectangle(frame, ( self.detection_x_start_position,self.detection_y_start_position), (self.detection_x_start_position+width, self.detection_y_start_position+height), border_color , 2)
            #  DRAW DETECTION BBOX 
            # for (x, y, w, h) in self.objects_in_detection_region:
            #     center_x=int(x+w/2)
            #     center_y=int(y+h/2)
            #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), int(2*self.video_resolution_ratio))
            #     cv2.circle(frame, (center_x,center_y), radius=2, color=(0, 255, 255), thickness=-1)

    def process_tracking_region(self,original_frame,frame,raw_detection_data,tracking_region):     
        self.current_tracked_objects=[]
        for dd in raw_detection_data:
            (x, y, w, h) = dd[0]
            center_x=int(x+w/2)
            center_y=int(y+h/2)
            if y+h>self.tracking_y_start_position+2 and center_y<self.tracking_y_end_position \
            and center_x>self.tracking_x_start_position and center_x<self.tracking_x_end_position :

                near_center_condidate=False
                if math.fabs(y-self.detection_y_start_position)<self.additional_marge_of_large_objects+1:
                    near_center_condidate=True
                bb_blob = original_frame[y:y+h, x:x+w]
                kps, des = self.sift.detectAndCompute(bb_blob, None)
                new_vehicle={ 'id':-1 ,'center_xy':(center_x,center_y), 'bbox_xywh':(x, y, w, h) ,'confidence':-1,'label':None, 'image':bb_blob,'key_points':kps,'description':des, 'region_blob':  SurveillanceRegionEnum.PRETRACKING_REGION if near_center_condidate else SurveillanceRegionEnum.TRACKING_REGION }
                self.current_tracked_objects.append(new_vehicle)
                
        self.assigne_newly_tracked_objects()
        self.assigne_tracked_objects()

        if self.debug_mode:
            height, width = tracking_region.shape[:2]
            colored_rect = np.zeros(tracking_region.shape, dtype=np.uint8) 
            colored_rect[:, :, 2] = 200
            border_color=(0,0,255)
            colored_tracking_region= cv2.addWeighted(tracking_region, 1, colored_rect, 0.5, 1.0)
            frame[self.tracking_y_start_position :self.tracking_y_end_position, self.tracking_x_start_position:self.tracking_x_end_position] = colored_tracking_region
            cv2.rectangle(frame, ( self.tracking_x_start_position,self.tracking_y_start_position), (self.tracking_x_start_position+width, self.tracking_y_start_position+height), border_color  , 2)
            #  DRAW TRACKING BBOX 
        
        for v in self.current_tracked_objects:
            (x, y, w, h)=v['bbox_xywh']
            center_x=int(x+w/2)
            center_y=int(y+h/2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), int(2*self.video_resolution_ratio))
            cv2.circle(frame, (center_x,center_y), radius=2, color=(255, 0, 255), thickness=-1)

    def init_regions(self,frame_size ): 
        width,height=frame_size[1],frame_size[0]
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
        # detection_section = np.zeros((self.debug_surveillance_section_height, frame.shape[1], 3), dtype=np.uint8)
        tracking_section = np.zeros((self.debug_surveillance_section_height, frame.shape[1], 3), dtype=np.uint8)
        cnn_section = np.zeros((self.debug_surveillance_section_height , frame.shape[1], 3), dtype=np.uint8)
        # detection_section[:,:,:] = (245, 227, 213)
        tracking_section[:,:,:] = (224, 223, 245)
        cnn_section[:,:,:] = (145, 227, 213)
        # self.add_detected_objects_to_debug_section(original_frame,detection_section)
        self.add_tracked_objects_to_debug_section(original_frame,tracking_section)
        self.add_cnn_objects_to_debug_section(cnn_section)
        # frame = cv2.vconcat([frame, detection_section])
        frame = cv2.vconcat([frame, tracking_section])
        frame = cv2.vconcat([frame, cnn_section])
        return frame

    def add_detected_objects_to_debug_section(self,original_frame,detection_section):
        offset=0
        for roi in self.objects_in_detection_region:
            (x, y, w, h) = roi 
            object = original_frame[y:y+h, x:x+w]
            if offset+w<detection_section.shape[1]:
                detection_section[:h,offset:w+offset] = object[:self.debug_surveillance_section_height,:]
            offset+=w

    def add_tracked_objects_to_debug_section(self,original_frame,tracking_section):
        offset=0
            #    for v in self.current_tracked_objects:
            # (x, y, w, h)=v['bbox_xywh']
        for v in self.current_tracked_objects:
            (x, y, w, h) = v['bbox_xywh']
            object = original_frame[y:y+h, x:x+w]
            if offset+w<tracking_section.shape[1]:
                tracking_section[:h,offset:w+offset] = object[:self.debug_surveillance_section_height,:]
            offset+=w

    def add_cnn_objects_to_debug_section(self,cnn_section):
        offset=0
        for cnn_object in self.DVR:
            (bbox_x,bbox_y,bbox_w,bbox_h)=cnn_object['bbox_xywh']

            if bbox_h>self.debug_surveillance_section_height-90:
                resized_crop=cv2.resize(cnn_object['image'], (self.debug_surveillance_section_height-90, self.debug_surveillance_section_height-90))
                bbox_h,bbox_w=resized_crop.shape[:2]
            else:
                resized_crop=cnn_object['image']

            if offset+resized_crop.shape[1]<cnn_section.shape[1]:

                region= cnn_object['region'].name 
                # cnn_section[:cnn_object['image'].shape[0],offset:cnn_object['image'].shape[1]+offset]=cnn_object['image'][:self.debug_surveillance_section_height,:]         
                cnn_section[:resized_crop.shape[0],offset:resized_crop.shape[1]+offset]=resized_crop[:self.debug_surveillance_section_height,:]
                # [cnn_bbox_y:min(self.debug_surveillance_section_height+cnn_bbox_y,cnn_bbox_h+cnn_bbox_y),cnn_bbox_x:cnn_bbox_x+cnn_bbox_w]
                cv2.putText(cnn_section, cnn_object['label'], ((offset ) , ( bbox_h) + 12 ), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25,25,250), 1)
                cv2.putText(cnn_section,  str(round(cnn_object['confidence'],2)), ((offset ) , ( bbox_h) + 28 ), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25,25,250), 1)
                # cv2.putText(cnn_section, str(len(cnn_object['key_points'])), ((offset ) , ( bbox_h) + 44 ), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (250,25,50), 2)
                cv2.putText(cnn_section,  region, ((offset ) , ( bbox_h) + 44 ), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (250,25,50), 2)
                cv2.putText(cnn_section, "ID : "+str(cnn_object['id']), ((offset ) , ( bbox_h) + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (250,125,50), 2)
                cv2.putText(cnn_section, str(cnn_object['status']), ((offset ) , ( bbox_h) + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (250,125,150), 2)
                cv2.putText(cnn_section, str(cnn_object['speed']), ((offset ) , ( bbox_h) + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (250,125,150), 2)
                
                offset+=bbox_w+35

    def detect_ROI_with_CNN(self,object): 
        if self.detection_service !=None and self.detection_service.get_selected_model() !=None:
            self.cnn_inference_count+=1
            print(f"CNN  inference_count : {self.cnn_inference_count}" )
            return  self.detection_service.detect_objects(object,boxes_plotting=False)[1]
        return []
    
    def addTrackingAndDetectionTimeAndFPS(self,img,detection_time,detection_fps):
        width=img.shape[1]
        cv2.rectangle(img,(int(width-195),10),(int(width-10),85),color=(240,240,240),thickness=-1)
        cv2.putText(img, f'FPS: {round(detection_fps,2)}', (int(width-190),30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (25,25,250), 2)
        cv2.putText(img, f'Det. time: {round(detection_time*1000)}ms', (int(width-190),52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (250,25,25), 2)
        # cv2.putText(img, f'Tra. time: {round(tracking_time*1000)}ms', (int(width-190),73), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (250,25,25), 2)


    # =============================== [TRACKING PROCESS] ==================================
    # =============(assigne_newly_tracked_objects , assigne_tracked_objects)===============

    def assigne_newly_tracked_objects(self):
        if len(self.DVR)==0:
            return
            
        # only_condidate_vehicles_to_tracked_from_DVR=[v for v in self.DVR_remaining_after_pretracked_assignment if v['status']=='PRE_TRACKED' or v['status']=='MISSED' ]
        only_condidate_vehicles_to_tracked_from_DVR=[v for v in self.DVR_remaining_after_pretracked_assignment  ]
        if len(only_condidate_vehicles_to_tracked_from_DVR)==0:
            return

        score_matrix=self.generate_matrix_of_scores(only_condidate_vehicles_to_tracked_from_DVR,self.current_tracked_objects)
        # self.remaining_current_tracked_objects=self.current_tracked_objects.copy()
        # self.DVR_remaining_after_newly_tracked_assignment=only_condidate_vehicles_to_tracked_from_DVR.copy()

        if len(score_matrix)>0:
            score_matrix=-np.array(score_matrix)
            row_ind, col_ind =linear_sum_assignment(score_matrix )
            for i in range(len(score_matrix) ):
                if (score_matrix[row_ind[i]][col_ind[i]]<self.INF ):
                    if ( score_matrix[row_ind[i]][col_ind[i]]<-self.similarity_threshold):
                        self.update_tracked_vehicle_properties(only_condidate_vehicles_to_tracked_from_DVR[col_ind[i]],self.current_tracked_objects[row_ind[i]])
       
    def assigne_tracked_objects(self):
        if len(self.DVR)==0:
            return
        only_vehicles_already_tracked_from_DVR=[v for v in self.DVR if v['status']=='TRACKED']
        if len(only_vehicles_already_tracked_from_DVR)==0:
            return

        # self.remaining_current_tracked_objects=self.current_tracked_objects.copy()
        self.DVR_remaining_after_tracked_assignment=only_vehicles_already_tracked_from_DVR.copy()

        score_matrix=self.generate_matrix_of_scores(only_vehicles_already_tracked_from_DVR,self.current_tracked_objects)
        if len(score_matrix)>0:
            score_matrix=-np.array(score_matrix)
            row_ind, col_ind =linear_sum_assignment(score_matrix )
            for i in range(len(score_matrix) ):
                if (score_matrix[row_ind[i]][col_ind[i]]<self.INF ):
                    if ( score_matrix[row_ind[i]][col_ind[i]]<-self.similarity_threshold):
                        registred_vehicle=only_vehicles_already_tracked_from_DVR[col_ind[i]]
                        current_vehicle= self.current_tracked_objects[row_ind[i]]
                        self.update_tracked_vehicle_properties(registred_vehicle,current_vehicle)
                        # self.remaining_current_tracked_objects.remove(next(filter(lambda vehicle: vehicle['center_xy'] == current_vehicle['center_xy'], self.remaining_current_tracked_objects)))
                        self.DVR_remaining_after_tracked_assignment.remove(next(filter(lambda vehicle: vehicle['id'] == registred_vehicle['id'], self.DVR_remaining_after_tracked_assignment)))
        self.process_exited_tracked_vehicles()
        
    
    # =========================== [DETECTION ZONE PROCESS] ================================
    # =============(assigne_pre_tracked_objects , assigne_detected_objects)===============

    def assigne_pre_tracked_objects(self):
        self.remaining_current_detected_objects_after_pretracking=[]
        if len(self.DVR)==0:
            # self.init_DVR_with_active_detections()
            return

        pre_tracked_vehicles_from_DVR=[v for v in self.DVR if v['status']=='PRE_TRACKED' ]
       
        self.remaining_current_detected_objects_after_pretracking=self.current_detected_objects.copy()
        self.DVR_remaining_after_pretracked_assignment=pre_tracked_vehicles_from_DVR.copy()

        score_matrix=self.generate_matrix_of_scores(pre_tracked_vehicles_from_DVR,self.current_detected_objects)
        if len(score_matrix)>0:
            score_matrix=-np.array(score_matrix)
            row_ind, col_ind =linear_sum_assignment(score_matrix )
            for i in range(len(score_matrix) ):
                if (score_matrix[row_ind[i]][col_ind[i]]<self.INF ):
                    if ( score_matrix[row_ind[i]][col_ind[i]]<-self.similarity_threshold):
                        registred_vehicle=pre_tracked_vehicles_from_DVR[col_ind[i]]
                        current_detected_vehicle=self.current_detected_objects[row_ind[i]]
                        self.remaining_current_detected_objects_after_pretracking.remove(next(filter(lambda vehicle: vehicle['center_xy'] == current_detected_vehicle['center_xy'], self.remaining_current_detected_objects_after_pretracking)))
                        self.DVR_remaining_after_pretracked_assignment.remove(next(filter(lambda vehicle: vehicle['id'] == registred_vehicle['id'], self.DVR_remaining_after_pretracked_assignment)))
                        self.update_detected_vehicle_properties(registred_vehicle,current_detected_vehicle,status='PRE_TRACKED')

    def assigne_detected_objects(self):
        self.remaining_current_detected_objects_after_detection=[]

        if len(self.DVR)==0 and len(self.current_detected_objects)>0:
            self.init_DVR_with_active_detections()
            # print("INIT DVR WHEN IS EMPTY")
            return

        if len(self.DVR_remaining_after_pretracked_assignment)==0 and len(self.remaining_current_detected_objects_after_pretracking)==0:
            return

        self.remaining_current_detected_objects_after_detection=self.current_detected_objects.copy()
        detected_missed_vehicles_from_DVR=[v for v in self.DVR if v['status']=='DETECTED' or v['status']=='MISSED' ]
        self.DVR_remaining_after_detectin_assignment=detected_missed_vehicles_from_DVR.copy()

        # # print("DVR_remaining_after_detectin_assignment")
        # # print( [str(obj['id'])+"-"+str(obj['center_xy']) for obj in  self.DVR_remaining_after_detectin_assignment])
        # print( "---------------")
        # print( "||| DVR_remaining_after_tracked_assignment :")
        # print( [str(obj['id'])+"-"+str(obj['center_xy']) for obj in self.DVR_remaining_after_tracked_assignment])
        
        # print( "DVR_remaining_after_pretracked_assignment :")
        # print( [str(obj['id'])+"-"+str(obj['center_xy']) for obj in self.DVR_remaining_after_pretracked_assignment])
        # print("remaining_current_detected_objects_after_pretracking :")
        # print( [str(obj['id'])+"-"+str(obj['center_xy']) for obj in self.remaining_current_detected_objects_after_pretracking])
        # print("current_detected_objects :")
        # print( [str(obj['id'])+"-"+str(obj['center_xy']) for obj in self.current_detected_objects])
        # print("detected_missed_vehicles_from_DVR :")
        # print( [str(obj['id'])+"-"+str(obj['center_xy']) for obj in detected_missed_vehicles_from_DVR])

        self.remaining_current_detected_objects_after_detection=self.remaining_current_detected_objects_after_pretracking.copy()
        score_matrix=self.generate_matrix_of_scores(detected_missed_vehicles_from_DVR,self.remaining_current_detected_objects_after_pretracking)
       
        # print("score_matrix")
        # print(score_matrix)
        # print("======")

        if len(score_matrix)>0:
            score_matrix=-np.array(score_matrix)
            row_ind, col_ind =linear_sum_assignment(score_matrix )
            for i in range(len(score_matrix) ):
                if (score_matrix[row_ind[i]][col_ind[i]]<self.INF ):
                    if ( score_matrix[row_ind[i]][col_ind[i]]<-self.similarity_threshold):
                        registred_vehicle=detected_missed_vehicles_from_DVR[col_ind[i]]
                        current_detected_vehicle=self.remaining_current_detected_objects_after_pretracking[row_ind[i]]
                        self.remaining_current_detected_objects_after_detection.remove(next(filter(lambda vehicle: vehicle['center_xy'] == current_detected_vehicle['center_xy'], self.remaining_current_detected_objects_after_detection)))
                        self.DVR_remaining_after_detectin_assignment.remove(next(filter(lambda vehicle: vehicle['id'] == registred_vehicle['id'], self.DVR_remaining_after_detectin_assignment)))
                        self.update_detected_vehicle_properties(registred_vehicle,current_detected_vehicle,status='DETECTED')
                        self.check_if_vehicle_enter_tracking_region(registred_vehicle)
                    else:
                        self.register_new_vehicle(self.remaining_current_detected_objects_after_pretracking[row_ind[i]])
                        # self.remaining_current_detected_objects_after_detection.remove(next(filter(lambda vehicle: vehicle['center_xy'] == current_detected_vehicle['center_xy'], self.remaining_current_detected_objects_after_detection)))
                        self.DVR_remaining_after_detectin_assignment.remove(next(filter(lambda vehicle: vehicle['id'] == registred_vehicle['id'], self.DVR_remaining_after_detectin_assignment)))
                        self.set_vehicle_state_as_missed(detected_missed_vehicles_from_DVR[col_ind[i]])

        self.update_DVR_remained_undetected_vehicles_state()
        self.register_remaining_current_detected_vehicles()

    def register_remaining_current_detected_vehicles(self):
        for v in self.remaining_current_detected_objects_after_detection:
            if v['is_a_large_vehicle']:
                if self.check_if_vehicle_in_tracking_zone(v):
                    continue
            self.DVR_vehicle_id+=1
            v['id']= self.DVR_vehicle_id
            self.colors[v['id']]=self.generateRandomTrackedColor()
            self.DVR.append( v )
            

    def set_vehicle_state_as_missed(self,registred_vehicle):
        registred_vehicle['status']='MISSED'
        registred_vehicle['missing_count']+=1

    def update_DVR_remained_undetected_vehicles_state(self):
        for v in self.DVR_remaining_after_detectin_assignment:
            remained_DVR_vehicle=(next(filter(lambda vehicle: vehicle['id'] == v['id'], self.DVR)))
            if remained_DVR_vehicle['status']=='TRACKED':
                self.check_if_vehicle_exit_tracking_region(remained_DVR_vehicle)
            # else:
            #     # self.set_registerd_vehicle_state(remained_DVR_vehicle,"MISSED")
            #     self.set_vehicle_state_as_missed(remained_DVR_vehicle)
            #     if remained_DVR_vehicle['missing_count']>5:
            #         self.DVR.remove(remained_DVR_vehicle)
            self.DVR_remaining_after_detectin_assignment.remove(v)

    def process_exited_tracked_vehicles(self):
        for v in self.DVR_remaining_after_tracked_assignment:
            if self.check_if_vehicle_exit_tracking_region(v):
                self.DVR_remaining_after_tracked_assignment.remove(v)



    def check_if_vehicle_in_tracking_zone(self,vehicle):
        tracked_vehicles_to_ignore=[v for v in self.DVR if v['status']=='TRACKED'   ]
        for v in tracked_vehicles_to_ignore:
            if self.calculate_similarity_score(vehicle,v)>self.similarity_threshold:
                return True
        return False

    def register_new_vehicle(self,vehicle): 
        # DO NOT registrate the new detected vehixle if is pretracked or TRACKED and its has a larger blob
    
        # if self.remaining_current_detected_objects:
        #     to_remove=next(filter(lambda v: v['center_xy'] == vehicle['center_xy'], self.remaining_current_detected_objects))
        #     self.remaining_current_detected_objects.remove(to_remove)

        tracked_vehicles_to_ignore=[v for v in self.DVR if v['status']=='TRACKED'   ]
        if vehicle['is_a_large_vehicle']:
            for v in tracked_vehicles_to_ignore:
                if self.calculate_similarity_score(vehicle,v)>self.similarity_threshold:
                    return False

        pre_vehicles_to_ignore=[v for v in self.DVR if v['status']=='PRE_TRACKED'   ]
        for v in pre_vehicles_to_ignore:
            similarity_score= self.calculate_similarity_score(vehicle,v)
            if similarity_score>self.similarity_threshold:
                return False
  
        self.DVR_vehicle_id+=1
        vehicle['id']= self.DVR_vehicle_id
        # vehicle['color_line']=self.generateRandomTrackedColor()
        self.colors[vehicle['id']]=self.generateRandomTrackedColor()
        self.DVR.append( vehicle )
        return True

    def init_DVR_with_active_detections(self):
        for v in self.current_detected_objects:
            self.DVR_vehicle_id+=1
            v['id']= self.DVR_vehicle_id
            self.colors[v['id']]=self.generateRandomTrackedColor()
            self.DVR.append( v )

    def update_tracked_vehicle_properties(self,registred_vehicle,active_vehicle):
        registred_vehicle['status']='TRACKED'
        registred_vehicle['bbox_xywh']=active_vehicle['bbox_xywh']
        registred_vehicle['image']=active_vehicle['image']
        registred_vehicle['key_points']=active_vehicle['key_points']
        registred_vehicle['description']=active_vehicle['description']
        registred_vehicle['center_xy']=active_vehicle['center_xy']
        registred_vehicle['speed']=self.calculate_vehicle_speed(registred_vehicle)
        self.center_pts[registred_vehicle['id']].append( registred_vehicle['center_xy'])
        registred_vehicle['tracklet_updated']=True
        # print( self.center_pts[registred_vehicle['id']])

    def update_detected_vehicle_properties(self,registred_vehicle,active_vehicle,status):
        registred_vehicle['status']=status
        registred_vehicle['bbox_xywh']=active_vehicle['bbox_xywh']
        registred_vehicle['image']=active_vehicle['image']
        registred_vehicle['key_points']=active_vehicle['key_points']
        registred_vehicle['description']=active_vehicle['description']
        registred_vehicle['center_xy']=active_vehicle['center_xy']
        registred_vehicle['confidence']=active_vehicle['confidence']
        registred_vehicle['label']=active_vehicle['label']
        registred_vehicle['speed']=self.calculate_vehicle_speed(registred_vehicle)
        self.center_pts[registred_vehicle['id']].append( registred_vehicle['center_xy'])
        registred_vehicle['tracklet_updated']=True


    def check_if_vehicle_exit_tracking_region(self,vehicle):
        _,y,_,h =vehicle['bbox_xywh']
        _,center_y =vehicle['center_xy']
        min_distance=150
        is_vehicle_in_image_last_border=math.fabs( y+h -self.tracking_y_end_position)<=1
        is_vehicle_center_is_lows=math.fabs(center_y-self.tracking_y_end_position)<min_distance
        if  is_vehicle_in_image_last_border and is_vehicle_center_is_lows :
            vehicle['status']='EXITED'
            return True
        return False

    def check_if_vehicle_enter_tracking_region(self,vehicle):
        x,y,w,h =vehicle['bbox_xywh']
        detection_tracking_transition_area_width=27
        if  math.fabs(y+h-self.detection_y_end_position)<detection_tracking_transition_area_width :
            vehicle['status']='PRE_TRACKED'
            return True
        return False

    def calculate_sift_similarity(self,active_vehicle,registred_vehicle):
        # start_time=time.perf_counter()
        epsilon=1e-4
        matches = self.sift_bf.knnMatch(active_vehicle['description'], registred_vehicle['description'], k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        matching_score = len(good_matches) / (len(matches)+epsilon)
        # print(time.perf_counter()-start_time)
        return matching_score

    def calculate_vehicle_speed(self,vehicle):
        speed=0
        return speed

    def calculate_similarity_score(self,active_vehicle,registred_vehicle):
        # active_vehicle_image=active_vehicle['image']
        # registred_vehicle_image=registred_vehicle['image']
        # registred_vehicle_image= cv2.resize(registred_vehicle_image, (active_vehicle_image.shape[1], active_vehicle_image.shape[0]))
        # ssim = structural_similarity(active_vehicle_image, registred_vehicle_image, multichannel=True,channel_axis=-1)
        # features = self.encoder(detection_frame, bboxes)
        sift_similarity=self.calculate_sift_similarity(active_vehicle,registred_vehicle)
        distance_score=self.euclidean_similarity(active_vehicle,registred_vehicle)
        return math.sqrt(distance_score)*sift_similarity

    def euclidean_similarity(self,vehicle_1,vehicle_2):
        center_x_1,center_y_1=vehicle_1['center_xy']
        center_x_2,center_y_2=vehicle_2['center_xy']
        score= 1/(math.fabs(center_x_1-center_x_2)+math.fabs(center_y_1-center_y_2)+1)
        return score
    
    def shape_similarity(self,vehicle_1,vehicle_2):
        x1,y1,w1,h1=vehicle_1['bbox_xywh']
        x2,y2,w2,h2=vehicle_2['bbox_xywh']
        score_w= (w1-w2) 
        score_h= (h1-h2) 
        if score_w+10<0 and score_h+10<0:
            return -1
        return score_w+score_h

    def reset(self):
        self.DVR=[]
        self.DVR_vehicle_id=0
        self.objects_in_detection_region=[]
        self.DVR_remaining_after_pretracked_assignment=[]
        self.DVR_remaining_after_detectin_assignment=[]
        self.DVR_remaining_after_tracked_assignment=[]
        self.current_tracked_objects=[]
        self.current_detected_objects=[]
        self.center_pts= [deque(maxlen=self.max_point) for _ in range(1000)]
        self.colors=[(255,255,255) for _ in range(1000)]

    def generate_matrix_of_scores(self,registered_vehicles_from_DVR,detected_objects):
        matrix_dim= max( len(registered_vehicles_from_DVR),len(detected_objects))
        score_matrix=[]
        for i in range(matrix_dim):
            score_array=[]
            if len(detected_objects)>i :
                for j in range(matrix_dim):
                    if len(registered_vehicles_from_DVR)>j:
                        score_array.append(self.calculate_similarity_score(detected_objects[i],registered_vehicles_from_DVR[j]))
                    else:
                        score_array.append(-self.INF)
            else:
                for j in range(matrix_dim):
                    score_array.append(-self.INF)
            score_matrix.append(score_array)
        # print('===============')
        # print( [str(obj['id'])+"-"+str(obj['center_xy']) for obj in self.active_detected_objects])
        # print( [str(obj['id'])+"-"+str(obj['center_xy']) for obj in non_tracked_vehicles_from_DVR])
        # print(score_matrix )
        return score_matrix

    def draw_tracking_points(self,frame):
        for index,vehicle_points_set in enumerate(self.center_pts):
            if len(vehicle_points_set)>0:
                for i in range(1,len(vehicle_points_set)):
                    thickness = int(np.sqrt(100/float(i+1))*1.5)
                    cv2.line(frame, (vehicle_points_set[i-1]), (vehicle_points_set[i]),  self.colors[index], thickness)

    def generateRandomTrackedColor(self):
        hue = random.random()
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        color= (int(r * 255), int(g * 255), int(b * 255))
        return color