
import colorsys
import random
import time
import cv2
import numpy as np
from utils_lib.enums import SurveillanceRegionEnum
from classes.background_subtractor_service import BackgroundSubtractorService

from classes.detection_service import IDetectionService
from utils_lib.utils_functions import addTrackingFrameFps

from utils_lib.deep_sort import preprocessing
from utils_lib.deep_sort import nn_matching
from utils_lib.deep_sort.detection import Detection
from utils_lib.deep_sort.tracker import Tracker
from _collections import deque

class TrackingService():
    track_object_by_background_sub=False
    track_object_by_cnn_detection=True
    # feature_detector = cv2.BRISK_create()
    show_missing_tracks=False
    pts = [deque(maxlen=30) for _ in range(1000)]
    background_subtractor_service:BackgroundSubtractorService=None    
    detection_service:IDetectionService=None    
    d_start,d_height,tr_start,tr_height= 200,100,300,500
    is_region_initialization_done=False   
    n_init=2
    max_age=30
    max_iou_distance=0.75
    max_cosine_distance = 0.7

    colors = {}

    def __init__(self,detection_service:IDetectionService,background_subtractor_service:BackgroundSubtractorService):
        self.background_subtractor_service=background_subtractor_service
        self.detection_service=detection_service
        nn_budget = None
        # nms_max_overlap = 0.8
        self.metric = nn_matching.NearestNeighborDistanceMetric('euclidean', self.max_cosine_distance, nn_budget)
        self.tracker = Tracker(self.metric,max_iou_distance=self.max_iou_distance, max_age=self.max_age, n_init=self.n_init)

    def apply(self,frame,threshold=0.5 ,nms_threshold=0.5): 

        start_time=time.perf_counter()
        detection_frame  ,raw_detection_data=self.getRawDetection(frame,threshold=threshold ,nms_threshold=nms_threshold)
        tracking_detections = [Detection(np.array(bbox),score, class_name, self.calculateFeatures(detection_frame,bbox)) for bbox, score, class_name in
                  raw_detection_data]
                
        self.tracker.predict()
        self.tracker.update(tracking_detections)

        for track in self.tracker.tracks:
            if (not track.is_confirmed() or track.time_since_update >1):
                if self.show_missing_tracks :
                    bbox = track.to_tlbr()
                    cv2.rectangle(detection_frame, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (255,255,255), 2)
                    cv2.putText(detection_frame, "?", (int(((bbox[0]) + (bbox[2]))/2)-10, int(((bbox[1]) + (bbox[3]))/2)), 0, 0.85,
                                (255, 255, 255), 2)  
            else:
                bbox = track.to_tlbr()
                class_name= track.get_class()
                color = self.getTrackedColor(int(track.track_id))
                # color = (32,122,200)
                cv2.rectangle(detection_frame, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 2)
                cv2.rectangle(detection_frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)
                            +len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(detection_frame, class_name+"-"+str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75,
                            (255, 255, 255), 2)        
                center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
                self.pts[track.track_id].append(center)

                for j in range(1, len(self.pts[track.track_id])):
                    if self.pts[track.track_id][j-1] is None or self.pts[track.track_id][j] is None:
                        continue
                    thickness = int(np.sqrt(64/float(j+1))*1.5)
                    cv2.line(detection_frame, (self.pts[track.track_id][j-1]), (self.pts[track.track_id][j]), color, thickness)

        try:
            fps=1/round(time.perf_counter()-start_time,3)
            addTrackingFrameFps(detection_frame,fps)
        except:
            addTrackingFrameFps(detection_frame,60)
        return detection_frame


    def getRawDetection(self,origin_frame,threshold=0.5 ,nms_threshold=0.5): 
        
        if self.track_object_by_cnn_detection==True and self.detection_service !=None and self.detection_service.get_selected_model() !=None:
            return  self.detection_service.detect_objects(origin_frame,threshold= threshold ,nms_threshold=nms_threshold,boxes_plotting=False)
        
        if self.track_object_by_background_sub==True and self.background_subtractor_service !=None :
            return  self.background_subtractor_service.apply(origin_frame,boxes_plotting=False)
        return origin_frame,[]


    def getTrackedColor(self,track_id):
        if self.colors.get(track_id):
            return self.colors.get(track_id)
        else:
            hue = random.random()
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            color= (int(r * 255), int(g * 255), int(b * 255))
            self.colors[track_id]=color
            return color
  
    def reset(self):
        self.tracker=Tracker(self.metric,max_iou_distance=self.max_iou_distance, max_age=self.max_age, n_init=self.n_init)
        self.pts = [deque(maxlen=30) for _ in range(1000)]
        self.colors={}
        # pass

    def  calculateFeatures (self,frame,bbox):
        x,y,w,h=bbox
        sub_frame=frame[y:y+h, x:x+w] 
        feature =np.array([])
        white_rect = np.zeros(sub_frame.shape, dtype=np.uint8) 
        white_rect[:, :, 1] = 255
        res = cv2.addWeighted(sub_frame, 0.5, white_rect, 0.2, 1.0)
        frame[y:y+h, x:x+w]=res
        return feature



        # def  calculateFeatures (self,frame,bbox):
        # x,y,w,h=bbox
        

        # # print(bbox)
        # sub_frame=frame[y:y+h, x:x+w] 
        # # hist1 = cv2.calcHist([sub_frame],[0],None,[256],[0,256])
        # # feature =np.array(hist1)
      
        # # feature =np.array([])
        # # feature =np.array(list(bbox))
        
        # gray = cv2.cvtColor(sub_frame, cv2.COLOR_BGR2GRAY)

        # # Detect keypoints in the image
        # # keypoints = self.feature_detector.detect(gray)
        # kp, descriptors = self.feature_detector.detectAndCompute(gray, None)
       
        # cv2.BFMatcher()
        # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # # Match the descriptors
        # matches = bf.match(descriptors, descriptors)
        # matches = sorted(matches, key=lambda x: x.distance)

        # if(matches and len(matches)>0):
        #     print(matches[0].distance)
        # # if(descriptors):
        # # print(list(np.array(descriptors).flat))
        # # print(descriptors.tolist())
        # # if(descriptors):
        # #     print(len(descriptors))

        # # feature =np.array(np.array(descriptors).flatten())
        # # feature =np.array(descriptors).flatten()
        # feature =np.array([])

        # white_rect = np.zeros(sub_frame.shape, dtype=np.uint8) 
        # white_rect[:, :, 1] = 255
        # res = cv2.addWeighted(sub_frame, 0.5, white_rect, 0.5, 1.0)
 
        # res = cv2.drawKeypoints(gray, kp, outImage=None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # frame[y:y+h, x:x+w]=res
        # # feature=np.array( )
        # return feature