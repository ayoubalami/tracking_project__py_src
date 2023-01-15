
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
    track_object_by_background_sub=True
    track_object_by_cnn_detection=False
    pts = [deque(maxlen=30) for _ in range(1000)]
    background_subtractor_service:BackgroundSubtractorService=None    
    detection_service:IDetectionService=None    
    d_start,d_height,tr_start,tr_height= 200,100,300,500
    is_region_initialization_done=False   

    colors = {}

    def __init__(self,detection_service:IDetectionService,background_subtractor_service:BackgroundSubtractorService):
        self.background_subtractor_service=background_subtractor_service
        self.detection_service=detection_service

        max_cosine_distance = 0.5
        nn_budget = None
        nms_max_overlap = 0.8
        metric = nn_matching.NearestNeighborDistanceMetric('euclidean', max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)

    def apply(self,frame,threshold=0.5 ,nms_threshold=0.5): 

        start_time=time.perf_counter()
        detection_frame  ,raw_detection_data=self.getRawDetection(frame,threshold=threshold ,nms_threshold=nms_threshold)

        tracking_detections = [Detection(np.array(bbox),score, class_name,np.array([])) for bbox, score, class_name in
                  raw_detection_data]
                
        self.tracker.predict()
        self.tracker.update(tracking_detections)

        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update >1:
                continue

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
        if self.track_object_by_background_sub==True and self.background_subtractor_service !=None :
            return  self.background_subtractor_service.apply(origin_frame,boxes_plotting=False)
        if self.track_object_by_cnn_detection==True and self.detection_service !=None and self.detection_service.get_selected_model() !=None:
            return  self.detection_service.detect_objects(origin_frame,threshold= threshold ,nms_threshold=nms_threshold,boxes_plotting=False)
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
  

