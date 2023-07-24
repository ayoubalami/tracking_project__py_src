
from collections import deque
import colorsys
import random
import time
import cv2
import numpy as np
from classes.detection_services.detection_service import IDetectionService
from utils_lib.enums import ObjectStatusEnum, SurveillanceRegionEnum
from classes.background_subtractor_service import BackgroundSubtractorService
from skimage.metrics import structural_similarity
import math
from scipy.optimize import linear_sum_assignment
from utils_lib.utils_functions import calculate_execution_time
import threading

class HybridTrackingService():
    
    background_subtractor_service:BackgroundSubtractorService=None
    detection_service:IDetectionService=None   

    # detection_y_start_ratio,detection_y_end_ratio=0.20,.30
    # tracking_y_start_ratio,tracking_y_end_ratio=0.30,1
    # detection_x_start_ratio,detection_x_end_ratio=0.2,.37
    # tracking_x_start_ratio,tracking_x_end_ratio=0.05,0.37

    detection_y_start_ratio,detection_y_end_ratio=0.32,.40
    tracking_y_start_ratio,tracking_y_end_ratio=0.40,1
    detection_x_start_ratio,detection_x_end_ratio=0.36,.52
    tracking_x_start_ratio,tracking_x_end_ratio=0.05,0.52
    # frame_shape=0
    additional_marge_of_large_objects=20

    is_region_initialization_done=False
    video_resolution_ratio=1
    objects_in_detection_region=[]
    DVR=[]
    DVR_vehicle_id=0
    # debug_surveillance_section=[]
    debug_surveillance_section_height=170
    debug_mode=False
    show_predicted_data=False

    max_timeout_of_missing=7
    x_padding=0
    y_padding=70
    sqrt_two=math.sqrt(2)

    missing_idx=[]
    global_missing_count=0
    global_tracked_count=0
    # orb = cv2.ORB_create()
    sift = cv2.SIFT_create() 
    sift_bf_matcher = cv2.BFMatcher()
   
    # frame_size=None
    detection_similarity_threshold=0.3
    tracking_similarity_threshold=0.15

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

    current_frame=0

    def __init__(self, detection_service:IDetectionService,background_subtractor_service:BackgroundSubtractorService  ):
        self.center_pts= [deque(maxlen=self.max_point) for _ in range(1000)]
        self.background_subtractor_service=background_subtractor_service
        self.detection_service=detection_service
        pass
    
    # @calculate_execution_time
    start_time_for_speed=0

    def apply(self,frame): 
        tracking_start_time=time.perf_counter()
        print(self.global_missing_count)
        print(self.global_tracked_count)
        if self.start_time_for_speed==0:
            self.start_time_for_speed=time.perf_counter()

        self.cnn_time=0
        if not self.is_region_initialization_done:
            self.frame_shape=frame.shape
            self.init_regions()
            self.is_region_initialization_done=True

        # bs_start_time=time.perf_counter()
        _  ,raw_detection_data=self.background_subtractor_service.apply(frame=frame,boxes_plotting=False)
        # bs_time=time.perf_counter()-bs_start_time

        original_frame=frame.copy()
        self.current_frame=frame

        detection_thread = threading.Thread(target=self.process_detection_region, args=(original_frame,raw_detection_data))  
        tracking_thread = threading.Thread(target=self.process_tracking_region, args=(original_frame,raw_detection_data))  

        detection_thread.start()
        tracking_thread.start()

        detection_thread.join()
        tracking_thread.join()

        # 
        # self.process_detection_region(original_frame,raw_detection_data)
        # self.process_tracking_region(original_frame,raw_detection_data)

        self.remove_exited_vehicles()

        self.draw_vehicle_bboxes(frame)
        self.draw_tracking_points(frame)

        if self.debug_mode:
            frame=self.add_debug_surveillance_section(original_frame,frame)
            detection_region , tracking_region=self.divide_draw_frame(frame)
            self.debug_results(frame,detection_region,tracking_region)
 
        self.addTrackingAndDetectionTimeAndFPS(frame,tracking_start_time)
               
        if self.cnn_time>0:
            self.dataTimeTrack.append(time.perf_counter()-tracking_start_time)
            print(f"Moyen Tracking_time :{sum(self.dataTimeTrack)/len(self.dataTimeTrack)} ")
            # self.dataTimeCNN.append(self.cnn_time)
            # self.dataTimeBS.append(bs_time)
       
            # print(f"Moyen CNN :{sum(self.dataTimeCNN)/len(self.dataTimeCNN)} ")
            # print(f"Moyen BS_time :{sum(self.dataTimeBS)/len(self.dataTimeBS)} ")
        
        return frame #inference_time


    dataTimePrepros=[]
    dataTimeTrack=[]
    dataTimeBS=[]
    dataTimeCNN=[]

    def divide_draw_frame(self,frame):
        detection_region= frame[self.detection_y_start_position:self.detection_y_end_position, self.detection_x_start_position:self.detection_x_end_position]
        tracking_region=  frame[self.tracking_y_start_position:self.tracking_y_end_position , self.tracking_x_start_position:self.tracking_x_end_position]
        return detection_region , tracking_region
    
    def init_regions(self ): 
        height,width=self.frame_shape[:2]
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

                ## region= cnn_object['region'].name 
                # cnn_section[:cnn_object['image'].shape[0],offset:cnn_object['image'].shape[1]+offset]=cnn_object['image'][:self.debug_surveillance_section_height,:]         
                cnn_section[:resized_crop.shape[0],offset:resized_crop.shape[1]+offset]=resized_crop[:self.debug_surveillance_section_height,:]
                # [cnn_bbox_y:min(self.debug_surveillance_section_height+cnn_bbox_y,cnn_bbox_h+cnn_bbox_y),cnn_bbox_x:cnn_bbox_x+cnn_bbox_w]
                cv2.putText(cnn_section, cnn_object['label'], ((offset ) , ( bbox_h) + 12 ), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25,25,250), 1)
                cv2.putText(cnn_section,  str(round(cnn_object['confidence'],2)), ((offset ) , ( bbox_h) + 28 ), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25,25,250), 1)
                # cv2.putText(cnn_section, str(len(cnn_object['key_points'])), ((offset ) , ( bbox_h) + 44 ), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (250,25,50), 2)
                # cv2.putText(cnn_section,  region, ((offset ) , ( bbox_h) + 44 ), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (250,25,50), 2)
                cv2.putText(cnn_section, "ID : "+str(cnn_object['id']), ((offset ) , ( bbox_h) + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (250,125,50), 2)
                cv2.putText(cnn_section, str(cnn_object['status'].name), ((offset ) , ( bbox_h) + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (250,125,150), 2)
                cv2.putText(cnn_section, str(format(cnn_object['speed'][0], ".2f")+" ; "+format(cnn_object['speed'][1], ".2f")), ((offset ) , ( bbox_h) + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (250,125,150), 2)
                offset+=bbox_w+35
    
    cnn_time=0
    def detect_ROI_with_CNN(self,object): 
        if self.detection_service !=None and self.detection_service.get_selected_model() !=None:
            self.cnn_inference_count+=1
            # print(f"CNN  inference_count : {self.cnn_inference_count}" )
            cnn_start_time=time.perf_counter()
            detection_results=  self.detection_service.detect_objects(object,boxes_plotting=False)[1]
            self.cnn_time=time.perf_counter()-cnn_start_time
            return detection_results
        return []
    
    def addTrackingAndDetectionTimeAndFPS(self,img,iteration_start_time):
        width=img.shape[1]
        process_time=time.perf_counter()-iteration_start_time
        if process_time>0:
            process_fps=1/process_time
        else :
            process_fps=0
        cv2.rectangle(img,(int(width-205),10),(int(width-10),85),color=(240,240,240),thickness=-1)
        cv2.putText(img, f'FPS: {round(process_fps,2)}', (int(width-200),30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (25,25,250), 2)
        cv2.putText(img, f'process dT: {round(process_time*1000)}ms', (int(width-200),52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (250,25,25), 2)
        # cv2.putText(img, f'Tra. time: {round(tracking_time*1000)}ms', (int(width-190),73), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (250,25,25), 2)

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
        self.background_subtractor_service.reset()

        self.dataTimePrepros=[]
        self.dataTimeCNN=[]
        self.dataTimeBS=[]
        self.dataTimeTrack=[]
   
    def generateRandomTrackedColor(self):
        hue = random.random()
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        color= (int(r * 255), int(g * 255), int(b * 255))
        return color

    def debug_results(self,frame,detection_region,tracking_region):

        height, width = detection_region.shape[:2]
        colored_rect = np.zeros(detection_region.shape, dtype=np.uint8)
        colored_rect[:, :, 0] = 200
        border_color=(255,0,0)
        colored_detection_region = cv2.addWeighted(detection_region, 1, colored_rect, 0.5, 1.0)
        frame[self.detection_y_start_position :self.detection_y_end_position, self.detection_x_start_position:self.detection_x_end_position] = colored_detection_region
        cv2.rectangle(frame, ( self.detection_x_start_position,self.detection_y_start_position), (self.detection_x_start_position+width, self.detection_y_start_position+height), border_color , 2)
        #############################        
        height, width = tracking_region.shape[:2]
        colored_rect = np.zeros(tracking_region.shape, dtype=np.uint8) 
        colored_rect[:, :, 2] = 200
        border_color=(0,0,255)
        colored_tracking_region= cv2.addWeighted(tracking_region, 1, colored_rect, 0.5, 1.0)
        frame[self.tracking_y_start_position :self.tracking_y_end_position, self.tracking_x_start_position:self.tracking_x_end_position] = colored_tracking_region
        cv2.rectangle(frame, ( self.tracking_x_start_position,self.tracking_y_start_position), (self.tracking_x_start_position+width, self.tracking_y_start_position+height), border_color  , 2)
        
    
    def draw_vehicle_bboxes(self,frame):
        
        for v in self.DVR:
            (x, y, w, h)=v['bbox_xywh']
            id=v['id']
            center_x,center_y=int(x+w/2),int(y+h/2)
            color,tickness,missed= ((0,0, 255),3,True ) if v['status']==ObjectStatusEnum.MISSED else ((0,255, 255) ,2,False )

            if missed:
                if y+h<self.frame_shape[0] :
                    # self.missing_idx.append(v['id'])
                    self.global_missing_count+= 1
                    # len(np.unique(np.array(self.missing_idx)))
            else:
                self.global_tracked_count+=1

            
            cv2.putText(frame, str(id), (x  ,  y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,color  , 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color,tickness)
            cv2.circle(frame, (center_x,center_y), radius=2, color=color, thickness=-1)

######################################################
    
    def process_tracking_region(self,original_frame,raw_detection_data):     
        self.current_tracked_objects=[]
        id=-1
        for dd in raw_detection_data:
            (x, y, w, h) = dd[0]
            center_x=int(x+w/2)
            center_y=int(y+h/2)
            
            if y+h>self.tracking_y_start_position  \
            and center_x>self.tracking_x_start_position and center_x<self.tracking_x_end_position :
                bb_blob = original_frame[y:y+h, x:x+w]
                
                kps, des = self.sift.detectAndCompute(bb_blob, None)
                new_vehicle={ 'id':id ,'center_xy':(center_x,center_y), 'bbox_xywh':(x, y, w, h),'confidence':-1,'label':None, 'image':bb_blob,'sift_key_points':kps,'sift_description':des,'region':SurveillanceRegionEnum.TRACKING_REGION ,'status':ObjectStatusEnum.TRACKED ,'missing_count':0,'speed':(0,0),'tracklet_updated':True,'timeout':0,'processed':False , 'center_xy_1':None,'last_detection_time':0,'predicted_center':(0,0),'predicted_bbox_xywh':None}
                id=id-1
                self.current_tracked_objects.append(new_vehicle)            
        self.assigne_tracked_objects()

    def process_detection_region(self,original_frame,raw_detection_data):        
        self.objects_in_detection_region=[]
        self.current_detected_objects=[]
        some_blobs_founded_in_detection_zone=False

        for dd in raw_detection_data:
            (x, y, w, h) = dd[0]
            center_x=int(x+w/2)
            center_y=int(y+h/2)
           
            if h+y>self.detection_y_start_position and y<self.detection_y_end_position \
            and center_x>self.detection_x_start_position and center_x<self.detection_x_end_position :
                some_blobs_founded_in_detection_zone=True
                break

        if some_blobs_founded_in_detection_zone:
            bb_blob = original_frame[self.detection_y_start_position -self.y_padding :self.detection_y_end_position +self.y_padding, self.detection_x_start_position:self.detection_x_end_position]
            cnn_detections=self.detect_ROI_with_CNN(bb_blob)
            # y_center_detection_zone=self.detection_y_start_position+((self.detection_y_end_position-self.detection_y_start_position)//2)

            for cnn_bbox,confidence,label in cnn_detections:
                x,y,w,h=cnn_bbox   
                y=y-self.y_padding      
                absolute_cnn_bbox_x,absolute_cnn_bbox_y =x+self.detection_x_start_position,y+self.detection_y_start_position

                if h+absolute_cnn_bbox_y<self.detection_y_start_position :
                    continue

                absolute_cnn_center_x,absolute_cnn_center_y=self.detection_x_start_position +x+(w//2),self.detection_y_start_position +y+(h//2)
                absolute_cnn_bbox_x,absolute_cnn_bbox_y =x+self.detection_x_start_position,y+self.detection_y_start_position
                cnn_crop=bb_blob[y+self.y_padding:min(self.debug_surveillance_section_height+y+self.y_padding,h+y+self.y_padding),x:x+w]
                kps, des = self.sift.detectAndCompute(cnn_crop, None)                
                new_vehicle={ 'id':-1 ,'center_xy':(absolute_cnn_center_x,absolute_cnn_center_y), 'bbox_xywh':(absolute_cnn_bbox_x,absolute_cnn_bbox_y,w ,h) ,'confidence':confidence,'label':label, 'image':cnn_crop,'sift_key_points':kps,'sift_description':des, 'region':SurveillanceRegionEnum.DETECTION_REGION ,'status':ObjectStatusEnum.DETECTED ,'missing_count':0,'speed':(0,0),'tracklet_updated':True,'timeout':0,'processed':False,'center_xy_1':None,'last_detection_time':0,'predicted_center':(0,0),'predicted_bbox_xywh':None}
                if h+absolute_cnn_bbox_y>=self.detection_y_end_position  :
                    # cv2.rectangle(frame, ( absolute_cnn_bbox_x, absolute_cnn_bbox_y), ( absolute_cnn_bbox_x+w, absolute_cnn_bbox_y+ h), (25,0,255)  , 3)
                    self.set_as_tracked_vehicle(new_vehicle)
                else:
                    # cv2.rectangle(frame, ( absolute_cnn_bbox_x, absolute_cnn_bbox_y), ( absolute_cnn_bbox_x+w, absolute_cnn_bbox_y+ h), (255,255,0)  , 2)
                    self.current_detected_objects.append(new_vehicle)

        self.assigne_detected_objects()

###############################################################

    def set_as_tracked_vehicle(self,active_vehicle):
       
        detected_missed_v_from_DVR=[v for v in self.DVR if v['status']==ObjectStatusEnum.DETECTED  ]
        scores = [(i,self.calculate_similarity_score(active_vehicle, registered_vehicle)) for (i,registered_vehicle) in enumerate(detected_missed_v_from_DVR)]
        if scores:
            max_index,max_score = max(scores, key=lambda x: x[1])  
            if max_score>self.detection_similarity_threshold:
                self.update_detected_vehicle_properties(detected_missed_v_from_DVR[max_index],active_vehicle,ObjectStatusEnum.TRACKED)
            pass

    def assigne_detected_objects(self):
        if len(self.DVR)==0 and len(self.current_detected_objects)>0:
            self.init_DVR_with_active_detections()
            return
        
        detected_missed_v_from_DVR=[v for v in self.DVR if v['status']==ObjectStatusEnum.DETECTED or v['status']==ObjectStatusEnum.MISSED ]
        self.remaining_current_detected_objects=self.current_detected_objects.copy()

        # print(">>>>>>> assigne_detected_objects")
        score_matrix=self.generate_matrix_of_scores(detected_missed_v_from_DVR,self.current_detected_objects,ObjectStatusEnum.DETECTED )
       
        if len(score_matrix)>0:
            score_matrix=-np.array(score_matrix)
            row_ind, col_ind =linear_sum_assignment(score_matrix )
            for i in range(len(score_matrix) ):
                if (score_matrix[row_ind[i]][col_ind[i]]<self.INF ):
                    current_detected_vehicle=self.current_detected_objects[row_ind[i]]
                    current_detected_vehicle['processed']=True
                    if ( score_matrix[row_ind[i]][col_ind[i]]<-self.detection_similarity_threshold):
                        registred_vehicle=detected_missed_v_from_DVR[col_ind[i]]
                        self.update_detected_vehicle_properties(registred_vehicle,current_detected_vehicle,status=ObjectStatusEnum.DETECTED)
                    else:
                        self.register_new_vehicle(self.remaining_current_detected_objects[row_ind[i]])

            # registre No registerd current_detected_objects
            [self.register_new_vehicle(i) for i in self.current_detected_objects if i['processed']==False ]
            for v in detected_missed_v_from_DVR :
                if v['processed']==False:
                    v['status']=ObjectStatusEnum.MISSED
                    v['missing_count']+=1

                v['processed']=False

    def init_DVR_with_active_detections(self):
        for v in self.current_detected_objects:
            self.DVR_vehicle_id+=1
            v['id']= self.DVR_vehicle_id
            self.colors[v['id']]=self.generateRandomTrackedColor()
            self.DVR.append( v )

#########################################################

    def init_DVR_with_active_trackers(self):
        for v in self.current_tracked_objects:
            self.DVR_vehicle_id+=1
            v['id']= self.DVR_vehicle_id
            self.colors[v['id']]=self.generateRandomTrackedColor()
            self.DVR.append( v )

    def update_tracked_vehicle_properties(self,registred_vehicle,active_vehicle):
        current_xy=active_vehicle['center_xy']
        previous_xy=registred_vehicle['center_xy_1']

        registred_vehicle['missing_count']=0
        registred_vehicle['center_xy_1']=registred_vehicle['center_xy']    
        registred_vehicle['status']=ObjectStatusEnum.TRACKED
        registred_vehicle['bbox_xywh']=active_vehicle['bbox_xywh']
        registred_vehicle['image']=active_vehicle['image']
        registred_vehicle['sift_key_points']=active_vehicle['sift_key_points']
        registred_vehicle['sift_description']=active_vehicle['sift_description']
        
        registred_vehicle['center_xy']=active_vehicle['center_xy']

        self.center_pts[registred_vehicle['id']].append( registred_vehicle['center_xy'])
        registred_vehicle['tracklet_updated']=True
        registred_vehicle['processed']=True
        registred_vehicle['speed']=self.calculate_vehicle_speed(previous_xy,current_xy)
        registred_vehicle['predicted_center'],registred_vehicle['predicted_bbox_xywh']= self.predict_next_position(registred_vehicle)
        
        if self.show_predicted_data:
            cv2.circle(self.current_frame, (int(registred_vehicle['predicted_center'][0]),int(registred_vehicle['predicted_center'][1])), radius=10, color=(0, 0, 255), thickness=-1)
            x,y,w,h=registred_vehicle['predicted_bbox_xywh']
            cv2.rectangle(self.current_frame, (x, y), (x + w, y + h),  (220, 220, 220), int(3))

    def assigne_tracked_objects(self):
      
        tracked_v_from_DVR=[v for v in self.DVR if v['status']==ObjectStatusEnum.TRACKED or v['status']==ObjectStatusEnum.MISSED ]
        score_matrix=self.generate_matrix_of_scores(tracked_v_from_DVR,self.current_tracked_objects,ObjectStatusEnum.TRACKED )
        
        if len(score_matrix)>0:
            score_matrix=-np.array(score_matrix)
            row_ind, col_ind =linear_sum_assignment(score_matrix )
            for i in range(len(score_matrix) ):
                # 
                if (score_matrix[row_ind[i]][col_ind[i]]<self.INF ):
                    current_tracked_vehicle=self.current_tracked_objects[row_ind[i]]
                    current_tracked_vehicle['processed']=True
                    if ( score_matrix[row_ind[i]][col_ind[i]]<-self.tracking_similarity_threshold):
                        registred_v=tracked_v_from_DVR[col_ind[i]]
                        registred_v['processed']=True
                        self.update_tracked_vehicle_properties(registred_v,current_tracked_vehicle)

        for v in tracked_v_from_DVR :
            if v['processed']==False:
                v['status']=ObjectStatusEnum.MISSED
                v['missing_count']+=1
            v['processed']=False
        
        pass

        # print(f"________")
    def remove_exited_vehicles(self):
        for v in [v for v in self.DVR]:
            if self.check_if_vehicle_exit_surveillance_region(v) or v['missing_count']>=10 :
                self.center_pts[v['id']]=deque(maxlen=self.max_point) 
                self.DVR.remove(v)
                continue
            # self.check_if_vehicle_is_missing(v)

    def check_if_vehicle_exit_surveillance_region(self,vehicle):
        _,y,_,h =vehicle['bbox_xywh']
        min_distance=70
        if math.fabs(y+h -self.tracking_y_end_position)<= 1 and (h<min_distance or vehicle['missing_count']>0):
            return True
        return False

    def generate_matrix_of_scores(self,vehicles_from_DVR,current_objects,object_status ):
        matrix_dim= max( len(vehicles_from_DVR),len(current_objects))
        score_matrix=[]
        for i in range(matrix_dim):
            score_array=[]
            if len(current_objects)>i :
                for j in range(matrix_dim):
                    if len(vehicles_from_DVR)>j:
                        score_array.append(self.calculate_similarity_score(current_objects[i],vehicles_from_DVR[j],object_status ))
                    else:
                        score_array.append(-self.INF)
            else:
                for j in range(matrix_dim):
                    score_array.append(-self.INF)
            score_matrix.append(score_array)
        
        # print(score_matrix)
        return score_matrix

    def calculate_similarity_score(self,active_vehicle,registred_vehicle,object_status=ObjectStatusEnum):
        sift_similarity=self.calculate_sift_similarity(active_vehicle,registred_vehicle)
        predictable_euclidean_similarity=self.predictable_euclidean_similarity(active_vehicle,registred_vehicle)
        predicted_iou=self.iou(active_vehicle,registred_vehicle)
        # surface_similarity=self.surface_similarity(active_vehicle,registred_vehicle)
        # color_hist_similarity=self.color_hist_similarity(active_vehicle,registred_vehicle)
        # w1,w2,w3=.15,.5,.35
       
        if object_status==ObjectStatusEnum.DETECTED:
            w=(.15,.4,.35,.1)
            # score=w[0]*predictable_euclidean_similarity**2 +w[1]*sift_similarity + w[2]*predicted_iou + w[3]*surface_similarity
        else:
            w=(.15,.4,.35,.1)
            # score=w[0]*predictable_euclidean_similarity**2 +w[1]*sift_similarity + w[2]*predicted_iou

        score=w[0]*predictable_euclidean_similarity**2 +w[1]*sift_similarity + w[2]*predicted_iou 

        # w=(.05,.55,.4)
        # if object_status==ObjectStatusEnum.DETECTED:
        #     score=w[0]*predictable_euclidean_similarity +w[1]*sift_similarity + w[2]*predicted_iou
        # else:
        #     score=w[0]*predictable_euclidean_similarity**2 +w[1]*sift_similarity + w[2]*predicted_iou

        # print("______ ")
        # print(self.calculate_sift_similarity(registred_vehicle,registred_vehicle))
        # print(self.euclidean_similarity(registred_vehicle,registred_vehicle))
        # print(self.iou(registred_vehicle,registred_vehicle))
        # print("")
        # print(f"{registred_vehicle['id'] }  -  {active_vehicle['id']}" )
        # print("__s")
        # print(predictable_euclidean_similarity)
        # # # print(euclidean_similarity)
        # print(sift_similarity)
        # print(predicted_iou)
        # # print(color_hist_similarity)
        # print("__f")

        # print(f"score {score}")
        # print(len(self.DVR))
        return score


    def calculate_sift_similarity(self,active_vehicle,registred_vehicle):
        # start_time=time.perf_counter()
        if len(active_vehicle['sift_key_points']) ==0 or len(registred_vehicle['sift_key_points'])==0:
            return 0
 
        sift_matches = self.sift_bf_matcher.knnMatch(active_vehicle['sift_description'], registred_vehicle['sift_description'], k=2)
        if len(sift_matches)==0:
            return 0

        sift_predicate_similarity = lambda match: len(match) == 2 and match[0].distance < 0.75 * match[1].distance
        good_matches = sum(1 for match in sift_matches if sift_predicate_similarity(match))
        return good_matches / (len(sift_matches))
        

    def predictable_euclidean_similarity(self,active_vehicle,registred_vehicle):
        center_x_1,center_y_1=active_vehicle['center_xy']
        center_x_2,center_y_2=registred_vehicle['center_xy']
        if registred_vehicle['predicted_center']!=(0,0):
            center_x_2,center_y_2=registred_vehicle['predicted_center']
 
        score= 1-((math.sqrt(( (center_x_1-center_x_2)/self.frame_shape[1] )**2+((center_y_1- center_y_2)/self.frame_shape[0])**2))/self.sqrt_two)
        # score= 1/(math.sqrt((center_x_1-center_x_2)**2+(center_y_1-center_y_2)**2)+1)
        return score
    
    # def euclidean_similarity(self,active_vehicle,registred_vehicle):
    #     center_x_1,center_y_1=active_vehicle['center_xy']
    #     center_x_2,center_y_2=registred_vehicle['center_xy'] 
        
    #     score= 1-((math.sqrt(( (center_x_1-center_x_2)/self.frame_shape[1] )**2+((center_y_1- center_y_2)/self.frame_shape[0])**2))/self.sqrt_two)
    #     # score= 1/(math.sqrt((center_x_1-center_x_2)**2+(center_y_1-center_y_2)**2)+1)
    #     return score

    def update_detected_vehicle_properties(self,registred_vehicle,active_vehicle,status):
        current_xy=active_vehicle['center_xy']
        previous_xy=registred_vehicle['center_xy_1']
        registred_vehicle['center_xy_1']=registred_vehicle['center_xy']
        registred_vehicle['status']=status
        registred_vehicle['bbox_xywh']=active_vehicle['bbox_xywh']
        registred_vehicle['image']=active_vehicle['image']
        # registred_vehicle['key_points']=active_vehicle['key_points']
        # registred_vehicle['description']=active_vehicle['description']
        registred_vehicle['sift_key_points']=active_vehicle['sift_key_points']
        registred_vehicle['sift_description']=active_vehicle['sift_description']

        registred_vehicle['center_xy']=active_vehicle['center_xy']
        registred_vehicle['confidence']=active_vehicle['confidence']
        registred_vehicle['label']=active_vehicle['label']
        self.center_pts[registred_vehicle['id']].append( registred_vehicle['center_xy'])
        registred_vehicle['tracklet_updated']=True
        registred_vehicle['processed']=True
        registred_vehicle['missing_count']=0
        registred_vehicle['speed']=self.calculate_vehicle_speed(previous_xy,current_xy)
        registred_vehicle['predicted_center'],registred_vehicle['predicted_bbox_xywh']= self.predict_next_position(registred_vehicle)        
        
        if self.show_predicted_data:
            cv2.circle(self.current_frame, (int(registred_vehicle['predicted_center'][0]),int(registred_vehicle['predicted_center'][1])), radius=6, color=(0, 255, 0), thickness=-1)
            x,y,w,h=registred_vehicle['predicted_bbox_xywh']
            cv2.rectangle(self.current_frame, (x, y), (x + w, y + h),  (0, 255, 0), int(2))
        
    def register_new_vehicle(self,vehicle): 
        self.DVR_vehicle_id+=1
        vehicle['id']= self.DVR_vehicle_id
        self.colors[vehicle['id']]=self.generateRandomTrackedColor()
        self.DVR.append( vehicle )
        return True

    def calculate_vehicle_speed(self,previous_xy,current_xy):
        if previous_xy==None:
            return (0,0)

        (x,y)=current_xy
        (x_1,y_1)=previous_xy       
        delta_time=time.perf_counter() - self.start_time_for_speed
        distance_y= (y-y_1)
        distance_x= (x-x_1)
        # if y_1>=y :
        #     distance_y*=-1
        # if x_1>=x :
        #     distance_x*=-1
        speed_x=round(distance_x/(delta_time+ 1e-5),4)
        speed_y=round(distance_y/(delta_time+ 1e-5),4)
        return speed_x,speed_y

    def predict_next_position(self,vehicle):
        def predict_next_center_xy(vehicle):
            speed_x,speed_y=vehicle['speed']
            delta_time=time.perf_counter() - self.start_time_for_speed
            self.start_time_for_speed=time.perf_counter()
            step_x= round(speed_x*delta_time,3)
            step_y= round(speed_y*delta_time,3)
            x,y=vehicle['center_xy']

            new_x=min(max(x+step_x , 0),self.frame_shape[1])
            new_y=min(max(y+step_y , 0),self.frame_shape[0])
           
            return (new_x,new_y ),( step_x,step_y)

        def predict_next_bbox(vehicle,new_step_xy):
            x,y,w,h=vehicle['bbox_xywh']            
            return  int(x+new_step_xy[0]),int(y+new_step_xy[1]),int(w),int(h)

        new_center,new_step_xy=predict_next_center_xy(vehicle)
        new_bbox=predict_next_bbox(vehicle,new_step_xy)
        return new_center,new_bbox
   
    def draw_tracking_points(self,frame):
        for index,vehicle_points_set in enumerate(self.center_pts):
            pts_set_size=len(vehicle_points_set)
            if pts_set_size>0:
                for i in range(1,pts_set_size):
                    # thickness = int(np.sqrt(100/float(i+1))*1.5)
                    thickness = int(18/math.sqrt(float(i+1)))
                    cv2.line(frame, (vehicle_points_set[i-1]), (vehicle_points_set[i]),  self.colors[index], thickness)

    def iou(self,active_vehicle,registred_vehicle):
        rect_1=active_vehicle["bbox_xywh"] 
        rect_2=registred_vehicle["predicted_bbox_xywh"]

        if rect_2==None :
            rect_2=registred_vehicle["bbox_xywh"]
 
        x1, y1, w1, h1 = rect_1
        x2, y2, w2, h2 = rect_2

        # Calculate coordinates of intersection
        x_intersection = max(x1, x2)
        y_intersection = max(y1, y2)
        w_intersection = min(x1 + w1, x2 + w2) - x_intersection
        h_intersection = min(y1 + h1, y2 + h2) - y_intersection
    
        # Check if there's no intersection
        if w_intersection <= 0 or h_intersection <= 0:
            return 0.0
        
        # Calculate areas of boxes and intersection
        area_box1 = w1 * h1
        area_box2 = w2 * h2
        area_intersection = w_intersection * h_intersection
        
        # Calculate IoU
        return (area_intersection) / float(area_box1 + area_box2 - area_intersection)
        # return iou

    # def surface_similarity(self,active_vehicle,registred_vehicle):
    #     rect_1=active_vehicle["bbox_xywh"] 
    #     rect_2=registred_vehicle["predicted_bbox_xywh"]

    #     if rect_2==None :
    #         rect_2=registred_vehicle["bbox_xywh"]
 
    #     _, _, w1, h1 = rect_1
    #     _, _, w2, h2 = rect_2
        
    #     surf_1=(w1*h1) 
    #     surf_2=(w2*h2) 

    #     surf_sim=0

    #     if surf_1>surf_2 :
    #         surf_sim= surf_2 / surf_1
        
    #     if surf_2>surf_1 :
    #         surf_sim= surf_1 / surf_2
        
    #     # print(f"surf : {surf_sim}")
    #     return surf_sim

    # def color_hist_similarity(self,active_vehicle,registred_vehicle):
    #     image_1=active_vehicle["image"] 
    #     image_2=registred_vehicle["image"]
    #     gray_image1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    #     gray_image2 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)

    #     histogram1 = cv2.calcHist([gray_image1], [0], None, [256], [0, 256])
    #     histogram2 = cv2.calcHist([gray_image2], [0], None, [256], [0, 256])
        
    #     # Normalize the histograms
    #     cv2.normalize(histogram1, histogram1, 0, 1, cv2.NORM_MINMAX)
    #     cv2.normalize(histogram2, histogram2, 0, 1, cv2.NORM_MINMAX)
    
    #     comparison_result = cv2.compareHist(histogram1, histogram2, cv2.HISTCMP_INTERSECT)
    #     return comparison_result
    #     # result = cv2.matchTemplate(input_image, template, cv2.TM_CCOEFF_NORMED)

