
import colorsys
from csv import writer
import random
import time
import cv2
import numpy as np
from utils_lib.enums import DetectorForTrackEnum
from utils_lib.enums import TrackingMethodEnum
from classes.background_subtractor_service import BackgroundSubtractorService
from classes.detection_services.detection_service import IDetectionService

from utils_lib.deep_sort import preprocessing
from utils_lib.deep_sort import nn_matching
from utils_lib.deep_sort.detection import Detection
from utils_lib.deep_sort.tracker import Tracker
from _collections import deque
from utils_lib.deep_sort.tools import generate_detections as gdet


# python app_controller.py -ds torch  -eval 1 -anch_nb 10 -skip_rate .0 -track_meth 2

class TrackingService():
    YOLO_FEATURE_EXTRACTOR=False
    tracker_detector=DetectorForTrackEnum.CNN_DETECTOR
    # feature_detector = cv2.BRISK_create()
    show_missing_tracks=False
    pts = [deque(maxlen=100) for _ in range(5000)]
    background_subtractor_service:BackgroundSubtractorService=None    
    detection_service:IDetectionService=None    
    # d_start,d_height,tr_start,tr_height= 200,100,300,500
    is_region_initialization_done=False
    n_init=3
    max_age=15
    threshold_feature_distance=0.2
    max_distance = 0.7
    feature_extractor_model_file='utils_lib/deep_sort/feature_extractors/mars-small128.pb'
    encoder=gdet.create_box_encoder(model_filename=feature_extractor_model_file,batch_size=8)

    # encoder=None
    colors = {}
    activate_detection_for_tracking=True
    mouse_tracked_coordinates=None
    raspberry_camera=None
    activate_camera_tracking=True
    frame_size=None

    # use_cnn_feature_extraction=True
    
    trackingMethodEnum:TrackingMethodEnum=TrackingMethodEnum.SORT
    skip_detection_rate=0
    save_time_results_csv=True
    # to rasp
    # tracked_object=None

    def __init__(self,detection_service:IDetectionService,background_subtractor_service:BackgroundSubtractorService,active_tracking_evaluation:False,trackingMethodEnum:TrackingMethodEnum,anchors_number,skip_detection_rate):
        self.background_subtractor_service=background_subtractor_service
        self.detection_service=detection_service
        nn_budget = None
        self.trackingMethodEnum=trackingMethodEnum
        self.anchors_number=anchors_number
        self.skip_detection_rate=skip_detection_rate
        # nms_max_overlap = 0.8
        # euclidean cosine
        self.metric = nn_matching.NearestNeighborDistanceMetric('cosine', self.threshold_feature_distance, nn_budget)
        self.active_tracking_evaluation=active_tracking_evaluation
        self.tracker = Tracker(self.metric,max_iou_distance=self.max_distance, max_age=self.max_age, n_init=self.n_init,trackingMethodEnum=self.trackingMethodEnum,active_tracking_evaluation=self.active_tracking_evaluation)
        self.file_name_time=str(time.perf_counter())
 
        print(self.TrackerDetails())


    def TrackerDetails(self):
        return f"Tracker with trackingMethodEnum : {self.trackingMethodEnum} ; anchors_number {self.anchors_number} ; active_tracking_evaluation : {self.active_tracking_evaluation} ; skip_detection_rate : {self.skip_detection_rate}"

    def apply(self,frame): 

        self.frame_size=frame.shape
        start_time=time.perf_counter()
        detection_data=self.getRawDetections(frame)
        detection_time=round(time.perf_counter()-start_time,3)

        # FOR TRACKING METHOD FOR OTHER DETECTION SERVICES OTHER THAN OPTIMIZED TOrCH
        tracking_time=time.perf_counter()
        if len(detection_data)==2:
            detection_frame  ,raw_detection_data=detection_data
            self.trackAndDrawBox(detection_frame,raw_detection_data)
        else:
            detection_frame,detected_objects,classesList=detection_data

            # print(self.TrackerDetails())
            processTime,detected_objects_count=self.optimized_trackAndDrawBox(detection_frame,detected_objects,classesList)
            self.process_times.append(processTime)
        
        tracking_time=round(time.perf_counter()-tracking_time,4)


        if self.save_time_results_csv:
            if detected_objects_count>0:
                csv_file = "csv_results/results_tracktime_"+ str(self.trackingMethodEnum)+"_"+ str(self.anchors_number)+"_"+self.file_name_time+".csv"
                with open(csv_file, 'a') as csvfile:
                    writer(csvfile).writerow([tracking_time,detected_objects_count])

        if round(time.perf_counter()-start_time,3)>0:
            tracking_fps=1/round(time.perf_counter()-start_time,3)
        else :
            tracking_fps=0
            
        if self.activate_camera_tracking and self.raspberry_camera :
            self.drawCenterLinesTarget(frame)
        #     self.raspberry_camera.tracked_object=self.returnSelectedTrackedObject(frame)

        # if self.raspberry_camera.tracked_object:
        #     print("raspberry_camera GO AFTER TRACKED : " +str(self.raspberry_camera.tracked_object.track_id)+" - "+self.raspberry_camera.tracked_object.to_tlwh())

        self.addTrackingAndDetectionTimeAndFPS(detection_frame,detection_time,tracking_time,tracking_fps)
        return detection_frame

    process_times=[]
    def optimized_trackAndDrawBox(self,frame,detected_objects,classesList):
        
        start_time=time.perf_counter() 
        bboxes=[]  
        scores=[]
        class_names=[]
        H , W=frame.shape[:2]
        x_ratio=self.detection_service.network_input_size/W
        y_ratio=self.detection_service.network_input_size/H
        for i  in range(len(detected_objects)):
            if len(detected_objects[i])==0:
                continue
            anchor,_ = detected_objects[i][0]
            x,y,w,h=anchor[1:5]
            classConfidence = anchor[5]
            classLabel = classesList[np.argmax(anchor[6:])]
            w=(int)(w/x_ratio)
            h=(int)(h/y_ratio)
            x=(int)(x/x_ratio-w/2)
            y=(int)(y/y_ratio-h/2)
  
            bboxes.append(np.array([x,y,w,h]))
            scores.append(classConfidence)
            class_names.append(classLabel)

        # if self.evaluate:
        #     # features = [np.array([]) for _  in bboxes] 

        #     # rate of disabling detection to simulate detection loses
        #     random_disable_detection=.25
        #     if random_disable_detection>0:
        #         random_number = random.random()
        #         if random_number <= random_disable_detection:
        #             # detected_objects=[]
        #             tracking_detections=[]
        #     else:
        #         # features = [np.array([]) for _  in bboxes] 
        #         features = self.encoder(frame, bboxes)
        #         tracking_detections = [Detection(bbox, score, class_name, feature=feature,object_data=object_data) for bbox, score, class_name, feature,object_data in
        #             zip(bboxes, scores, class_names, features,detected_objects)]
        # else:
        if self.active_tracking_evaluation:
            # random_disable_detection=.0
            # if random_disable_detection>0:
            random_number = random.random()
            if random_number >= self.skip_detection_rate:
                features = self.encoder(frame, bboxes)
                tracking_detections = [Detection(bbox, score, class_name, feature=feature,object_data=object_data) for bbox, score, class_name, feature,object_data in
                    zip(bboxes, scores, class_names, features,detected_objects)]
                self.tracker.update(tracking_detections,W=W,H=H, x_ratio=x_ratio, y_ratio=y_ratio)

        else:
            if self.trackingMethodEnum==TrackingMethodEnum.DEEP_SORT:
                features = self.encoder(frame, bboxes)
                tracking_detections = [Detection(bbox, score, class_name, feature=feature) for bbox, score, class_name, feature in
                    zip(bboxes, scores, class_names, features)]
            elif self.trackingMethodEnum==TrackingMethodEnum.SORT:
                features = [np.array([]) for _  in bboxes] 
                tracking_detections = [Detection(bbox, score, class_name, feature=feature) for bbox, score, class_name, feature in
                    zip(bboxes, scores, class_names, features)]
            elif self.trackingMethodEnum==TrackingMethodEnum.ANCHOR_BASED:
                tracking_detections = [Detection(bbox, score, class_name,object_data=object_data) for bbox, score, class_name,  object_data in
                    zip(bboxes, scores, class_names, detected_objects)]

            self.tracker.update(tracking_detections,W=W,H=H, x_ratio=x_ratio, y_ratio=y_ratio)

            # if self.use_cnn_feature_extraction:
            #     # features = self.encoder(frame, bboxes)
            #     features = [np.array([]) for _  in bboxes] 
            #     tracking_detections = [Detection(bbox, score, class_name, feature=feature) for bbox, score, class_name, feature in
            #         zip(bboxes, scores, class_names, features)]
            # else:
            #     tracking_detections = [Detection(bbox, score, class_name,object_data=object_data) for bbox, score, class_name,  object_data in
            #         zip(bboxes, scores, class_names, detected_objects)]


        self.tracker.predict()  
        tracking_time=time.perf_counter()-start_time

        self.drawTrackedBBoxes(frame)
        self.drawDetectedBBoxes(frame,bboxes)
        return tracking_time,len(detected_objects)
        # print("==optimized_trackAndDrawBox")
        # pass
 
    def trackAndDrawBox(self,detection_frame,raw_detection_data):
        H,W=detection_frame.shape[:2]
        bboxes = [np.array(raw_data[0]) for raw_data  in raw_detection_data]       
        scores = [raw_data[1] for raw_data  in raw_detection_data]       
        class_names = [raw_data[2] for raw_data  in raw_detection_data]       

        if self.trackingMethodEnum==TrackingMethodEnum.DEEP_SORT:
            features = self.encoder(detection_frame, bboxes)
        if self.trackingMethodEnum==TrackingMethodEnum.SORT:
            features=  [np.array([]) for _  in bboxes] 

        tracking_detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                  zip(bboxes, scores, class_names, features)]
        self.tracker.update(tracking_detections,H,W)
        self.tracker.predict()
        self.drawTrackedBBoxes(detection_frame)
        self.drawDetectedBBoxes(detection_frame,bboxes)

    def getRawDetections(self,origin_frame): 
        if self.activate_detection_for_tracking:
            if self.tracker_detector==DetectorForTrackEnum.CNN_DETECTOR and self.detection_service !=None and self.detection_service.get_selected_model() !=None:
                return  self.detection_service.detect_objects(origin_frame,boxes_plotting=False)
            if self.tracker_detector==DetectorForTrackEnum.BACKGROUND_SUBTRACTION  and self.background_subtractor_service !=None :
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
        self.tracker.tracks=[]
        self.tracker=Tracker(self.metric,max_iou_distance=self.max_distance, max_age=self.max_age, n_init=self.n_init,trackingMethodEnum=self.trackingMethodEnum,active_tracking_evaluation=self.active_tracking_evaluation)
        self.pts = [deque(maxlen=30) for _ in range(10000)]
        self.colors={}
        # pass

    def  drawDetectedBBoxes (self,frame,bboxes):
        for bbox in bboxes:
            x,y,w,h=bbox
            sub_frame=frame[y:y+h, x:x+w] 
            white_rect = np.zeros(sub_frame.shape, dtype=np.uint8) 
            white_rect[:, :, 1] = 200
            res = cv2.addWeighted(sub_frame, 0.5, white_rect, 0.2, 1.0)
            try :
                frame[y:y+h, x:x+w]=res
            except:
                pass
            # cv2.rectangle(frame, ( x,y), (x+w,y+h), (255,255,255), 3)

    def drawTrackedBBoxes(self,frame):
        for track in self.tracker.tracks:
            if (not track.is_confirmed() or track.time_since_update >1):
                if self.show_missing_tracks:
                    bbox = track.to_tlbr()
                    cv2.rectangle(frame, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (255,255,255), 2)
                    cv2.putText(frame, "?", (int(((bbox[0]) + (bbox[2]))/2)-10, int(((bbox[1]) + (bbox[3]))/2)), 0, 0.85,
                                (255, 255, 255), 2)  
            else:
                bbox = track.to_tlbr()
                # bbox = track.detection_bbox.to_tlbr()

                class_name= track.get_class()
                color = self.getTrackedColor(int(track.track_id))
                
                cv2.rectangle(frame, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-20)), (int(bbox[0])+(len(class_name)
                            +len(str(track.track_id)))*13, int(bbox[1])), color, -1)
                cv2.putText(frame, class_name+"-"+str(track.track_id), (int(bbox[0]), int(bbox[1]-3)), 0, 0.50,
                            (255, 255, 255), 2)        
                center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
                self.pts[track.track_id].append(center)

                for j in range(1, len(self.pts[track.track_id])):
                    if self.pts[track.track_id][j-1] is None or self.pts[track.track_id][j] is None:
                        continue
                    thickness = max(int(7-(j//15)),2)
                    cv2.line(frame, (self.pts[track.track_id][j-1]), (self.pts[track.track_id][j]), color, thickness)

    # def goToTrackedPosition(self,frame):
    #     self.drawCenterLinesTarget(frame)
    #     x,y = self.tracked_coordinates
    #     if x and y:
    #         heigth,width=frame.shape[:2]
    #         x=round(x*width)
    #         y=round(y*heigth)
    #         print("Go to "+str(x)+' '+str(y))
    #         self.raspberry_camera.moveServoMotorToCoordinates(origins=(width,heigth),destination_coordinates=(x,y),speed=0.009)
    #         self.tracked_coordinates=(None,None)

    def drawCenterLinesTarget(self,frame):
        heigth,width=frame.shape[:2]
        
        line_size=75
        thickness=7
        if self.raspberry_camera:
            line_size=75 if self.raspberry_camera.zoom==1 else 75/self.raspberry_camera.zoom
            thickness=7 if self.raspberry_camera.zoom==1 else int(7/self.raspberry_camera.zoom)

        cv2.line(frame, (  int(width/2 - line_size),int(heigth/2)), (  int(width/2 + line_size),int(heigth/2)),  ( 0,255, 0),  thickness, cv2.LINE_AA)
        cv2.line(frame, (  int(width/2),int(heigth/2-line_size)), (  int(width/2 ),int(heigth/2+line_size)),  ( 0,255, 0),  thickness, cv2.LINE_AA)
       
    def addTrackingAndDetectionTimeAndFPS(self,img,detection_time,tracking_time,tracking_fps):
        width=img.shape[1]
        cv2.rectangle(img,(int(width-195),10),(int(width-10),85),color=(240,240,240),thickness=-1)
        cv2.putText(img, f'FPS: {round(tracking_fps,2)}', (int(width-190),30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (25,25,250), 2)
        cv2.putText(img, f'Det. time: {round(detection_time*1000)}ms', (int(width-190),52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (250,25,25), 2)
        cv2.putText(img, f'Tra. time: {round(tracking_time*1000)}ms', (int(width-190),73), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (250,25,25), 2)
 
    def returnSelectedTrackedObject(self,coordinates):
        self.mouse_tracked_coordinates=coordinates
        if self.mouse_tracked_coordinates:
            x,y = self.mouse_tracked_coordinates
            heigth,width=self.frame_size[:2]
            print(heigth,width)
            print( x,y)
            x=round(x*width)
            y=round(y*heigth)
            
            for track in self.tracker.tracks:
                if ( track.is_confirmed() and track.time_since_update <=1):
                    # FONCTION ERROR
                    (l,t,r,b) = track.to_tlbr()
                    if y>=t and y<=b and x>=l and x<=r:
                        self.mouse_tracked_coordinates=None
                        print("AN OBJECT TO TRACK IS FOUNDED : " +str(track.track_id)+" - "+str(track.to_tlwh()) )
                        return track
            self.mouse_tracked_coordinates=None
            print("NO OBJECT FOUNDED")
            return None
        
        # if self.raspberry_camera.tracked_object and self.raspberry_camera.tracked_object.is_confirmed():
        #     print("CONTINUE TRACKING  - 30 TTL : "+str(self.raspberry_camera.tracked_object.time_since_update )) 
        #     return self.raspberry_camera.tracked_object





















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