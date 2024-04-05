
# import colorsys
# import random
# import time
# from classes.tracking_service.sort import Sort
# import cv2
# import numpy as np
# from utils_lib.enums import DetectorForTrackEnum
# from classes.background_subtractor_service import BackgroundSubtractorService
# from classes.detection_services.detection_service import IDetectionService

# from utils_lib.deep_sort import preprocessing
# from utils_lib.deep_sort import nn_matching
# from utils_lib.deep_sort.detection import Detection
# from utils_lib.deep_sort.tracker import Tracker
# from _collections import deque
# # from utils_lib.deep_sort.tools import generate_detections as gdet
# from scipy.spatial.distance import cosine
# from scipy.optimize import linear_sum_assignment
# from statistics import mode

# class OptimizedTrackingService():
#     tracker_detector=DetectorForTrackEnum.CNN_DETECTOR
#     # feature_detector = cv2.BRISK_create()
#     show_missing_tracks=False
#     pts = [deque(maxlen=50) for _ in range(10000)]
#     background_subtractor_service:BackgroundSubtractorService=None    
#     detection_service:IDetectionService=None    
#     d_start,d_height,tr_start,tr_height= 200,100,300,500
#     is_region_initialization_done=False   
#     n_init=3
#     max_age=5
#     threshold_feature_distance=0.2
#     max_distance = 0.7
#     feature_extractor_model_file='utils_lib/deep_sort/feature_extractors/mars-small128.pb'
#     # encoder=gdet.create_box_encoder(model_filename=feature_extractor_model_file,batch_size=1)
#     encoder=None
#     colors = {}
#     use_cnn_feature_extraction=False
#     activate_detection_for_tracking=True
#     mouse_tracked_coordinates=None
#     raspberry_camera=None
#     activate_camera_tracking=True
#     frame_size=None
#     # to rasp
#     # tracked_object=None

#     def __init__(self,detection_service:IDetectionService,background_subtractor_service:BackgroundSubtractorService):
#         self.background_subtractor_service=background_subtractor_service
#         self.detection_service=detection_service
#         # nn_budget = None
#         # nms_max_overlap = 0.8
#         # euclidean
#         # self.metric = nn_matching.NearestNeighborDistanceMetric('cosine', self.threshold_feature_distance, nn_budget)
#         # self.tracker = Tracker(self.metric,max_iou_distance=self.max_distance, max_age=self.max_age, n_init=self.n_init)
#         self.mot_tracker = Sort(max_age=self.max_age, 
#                                min_hits=3,
#                                iou_threshold=0.3)
#                                 #create instance of the SORT tracker
#         #   
#     def apply(self,frame): 
#         start_time=time.perf_counter()
#         self.frame_size=frame.shape 
#         detection_frame  ,raw_detection_data,clusters,features=self.getRawDetections(frame)
#         detection_time=round(time.perf_counter()-start_time,3)
#         # print(raw_detection_data)
#         tracking_time=time.perf_counter()

#         # self.optimized_trackAndDrawBox(detection_frame,raw_detection_data)
#         self.YOLOFeatureTrack(frame,clusters,features)

#         tracking_time=round(time.perf_counter()-tracking_time,4)

#         if round(time.perf_counter()-start_time,3)>0:
#             tracking_fps=1/round(time.perf_counter()-start_time,3)
#         else :
#             tracking_fps=0

#         if self.activate_camera_tracking and self.raspberry_camera :
#             self.drawCenterLinesTarget(frame)
#         #     self.raspberry_camera.tracked_object=self.returnSelectedTrackedObject(frame)

#         # if self.raspberry_camera.tracked_object:
#         #     print("raspberry_camera GO AFTER TRACKED : " +str(self.raspberry_camera.tracked_object.track_id)+" - "+self.raspberry_camera.tracked_object.to_tlwh())

#         self.addTrackingAndDetectionTimeAndFPS(detection_frame,detection_time,tracking_time,tracking_fps)
#         return detection_frame

#     trackers=None
#     # trackers=[]
#     def YOLOFeatureTrack(self,frame,clusters,features):
 
#         # S_clusters,M_clusters,L_clusters=clusters
#         # S_features,M_features,L_features=features
#         detections=(clusters,features)
#         # print(len(clusters))
#         # print(len(features))
#         # print(clusters.shape)
#         # print(features.shape)

#         if self.trackers==None:
#             self.trackers=detections
#             # print(self.trackers==None)
#             pass

#         matrix_res=self.object_similarity(detections,detections)
#         self.trackers=detections
#         print(matrix_res)
        
#         results=[]

#         for i in range(len(matrix_res)):
#             if len(matrix_res[i])==0:
#                 continue
#             row_indices, col_indices =linear_sum_assignment(matrix_res[i])
#             results.append((row_indices, col_indices))
#             print("Assigned Row Indices:", row_indices)
#             print("Assigned Column Indices:", col_indices)
#             print("===")
#             print("===")
#         # most_frequent_result = mode(results)
#         most_frequent_result = np.array(results).mean(axis=0).round().astype(int)

#         print("Most frequent result:")
#         print( most_frequent_result)


#     def object_similarity(self,detections,trackers):
#         # (clusters_small_1,clusters_medium_1,clusters_large_1),(S_features_1,M_features_1,L_features_1)=detections
#         # (clusters_small_2,clusters_medium_2,clusters_large_2),(S_features_2,M_features_2,L_features_2)=trackers
#         clusters_1,features_1=detections
#         clusters_2,features_2=trackers

#         threshold=0.2
#         somme=0
#         matrix_res=[]
#         # matrix_avg=[]
#         # matrix_min=[]
#         # matrix_max=[]

#         for I in range(1):
#             I=0
#             matrix_avg=[]
#             matrix_min=[]
#             matrix_max=[]
#             for i in range(len(clusters_1[I])):
#                 vect_avg=[]
#                 vect_min=[]
#                 vect_max=[]
#                 for j in range(len(clusters_2[I])):
#                     vector_distances=[[self.vector_distance(x[0],features_1[I],y[0],features_2[I]) for y in clusters_2[I][j]] for x in clusters_1[I][i]]
#                     vect_avg.append(np.mean(vector_distances))
#                     vect_max.append(np.max(vector_distances))
#                     vect_min.append(np.min(vector_distances))

#                 matrix_avg.append(vect_avg)
#                 matrix_max.append(vect_max)
#                 matrix_min.append(vect_min)
#             matrix_res.append(matrix_max)
#             matrix_res.append(matrix_min)
#             matrix_res.append(matrix_avg)
#         return matrix_res   
         
#         # self.drawTrackedBBoxes(trackers,frame)
#     def from_id_to_featureMap(self,id):
#         self.detection_service.network_input_size


#         if id < 8112:
#             y=(id/3/52)
#             x=(id/3)%52
#         elif id < 10140 and 8112 <= id:
#             id=id-8112
#             y=(id/3/26)
#             x=(id/3)%26
#         elif id >= 10140:
#             id=id-10140
#             y=(id/3/13)
#             x=(id/3)%13
#         return (int)(x),(int)(y)

#     def vector_distance(self,idx_1,features_1,idx_2,features_2):
#         col_1,line_1=self.from_id_to_featureMap(idx_1)
#         col_2,line_2=self.from_id_to_featureMap(idx_2)

#         return cosine( features_1[0][line_1][col_1],features_2[0][line_2][col_2])
        
#     def optimized_trackAndDrawBox(self,frame,raw_detection_data):
#         S_raw_detection_data,M_raw_detection_data,L_raw_detection_data=raw_detection_data

#         index = [raw_data[0] for raw_data  in S_raw_detection_data]       
#         bboxes = [np.array(raw_data[1]) for raw_data  in S_raw_detection_data]       
#         scores = [raw_data[2] for raw_data  in S_raw_detection_data]       
#         class_names = [str(raw_data[3]) for raw_data  in S_raw_detection_data]       
#         if self.use_cnn_feature_extraction:
#             # features = self.encoder(detection_frame, bboxes)
#             features = [np.array([]) for _  in bboxes] 
#         else:
#             features=  [np.array([]) for _  in bboxes] 

#         tracking_detections=np.array([[*bbox,score] for bbox, score in  zip(bboxes, scores)])
#         # tracking_detections[:, 2:4] += dets[:, 0:2]
#         # tracking_detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
#         #           zip(bboxes, scores, class_names, features)]
#         # print(tracking_detections)

#         trackers = self.mot_tracker.update(tracking_detections)

#         # print(len(trackers))
#         # print(trackers)

#         # self.tracker.update(tracking_detections)
#         # self.tracker.predict()
#         self.drawTrackedBBoxes(trackers,frame)
#         # self.drawDetectedBBoxes(frame,bboxes)

     

#     def getRawDetections(self,origin_frame): 
#         if self.activate_detection_for_tracking:
#             if self.tracker_detector==DetectorForTrackEnum.CNN_DETECTOR and self.detection_service !=None and self.detection_service.get_selected_model() !=None:
#                 return  self.detection_service.detect_objects(origin_frame,boxes_plotting=False)
#             if self.tracker_detector==DetectorForTrackEnum.BACKGROUND_SUBTRACTION  and self.background_subtractor_service !=None :
#                 return  self.background_subtractor_service.apply(origin_frame,boxes_plotting=False)
#         return origin_frame,[]

#     def getTrackedColor(self,track_id):
#         if self.colors.get(track_id):
#             return self.colors.get(track_id)
#         else:
#             hue = random.random()
#             r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
#             color= (int(r * 255), int(g * 255), int(b * 255))
#             self.colors[track_id]=color
#             return color
  
#     def reset(self):
#         self.tracker.tracks=[]
#         self.tracker=Tracker(self.metric,max_iou_distance=self.max_distance, max_age=self.max_age, n_init=self.n_init)
#         self.pts = [deque(maxlen=30) for _ in range(10000)]
#         self.colors={}
#         # pass

#     def  drawDetectedBBoxes (self,frame,bboxes):
#         for bbox in bboxes:
#             x,y,w,h=bbox
#             sub_frame=frame[y:y+h, x:x+w] 
#             white_rect = np.zeros(sub_frame.shape, dtype=np.uint8) 
#             white_rect[:, :, 1] = 200
#             res = cv2.addWeighted(sub_frame, 0.5, white_rect, 0.2, 1.0)
#             frame[y:y+h, x:x+w]=res
#             # cv2.rectangle(frame, ( x,y), (x+w,y+h), (255,255,255), 3)

#     def drawTrackedBBoxes(self,trackers,frame):
#         for track in trackers:
#             # if (not track.is_confirmed() or track.time_since_update >1):
#             #     if self.show_missing_tracks:
#             #         bbox = track.to_tlbr()
#             #         cv2.rectangle(frame, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (255,255,255), 2)
#             #         cv2.putText(frame, "?", (int(((bbox[0]) + (bbox[2]))/2)-10, int(((bbox[1]) + (bbox[3]))/2)), 0, 0.85,
#             #                     (255, 255, 255), 2)  
#             # else:
#             #     bbox = track.to_tlbr()
#                 # bbox = track.detection_bbox.to_tlbr()
                
#                 x,y,X,Y,track_id=track
#                 track_id=int(track_id)
#                 color = self.getTrackedColor(track_id)
                
#                 cv2.rectangle(frame, (int(x),int(y)), (int(X),int(Y)), color, 2)
#                 cv2.rectangle(frame, (int(x), int(y-20)), (int(X), int(y)), color, -1)
#                 cv2.putText(frame, "-"+str(track_id), (int(x), int(y-5)), 0, 0.50,
#                             (255, 255, 255), 2)        
#                 center = int(x + (X-x)/2), int((y+(Y-y)/2))
#                 self.pts[track_id].append(center)

#                 for j in range(1, len(self.pts[track_id])):
#                     if self.pts[track_id][j-1] is None or self.pts[track_id][j] is None:
#                         continue
#                     thickness = int(np.sqrt(64/float(j+1))*1.5)
#                     cv2.line(frame, (self.pts[track_id][j-1]), (self.pts[track_id][j]), color, thickness)

#     # def goToTrackedPosition(self,frame):
#     #     self.drawCenterLinesTarget(frame)
#     #     x,y = self.tracked_coordinates
#     #     if x and y:
#     #         heigth,width=frame.shape[:2]
#     #         x=round(x*width)
#     #         y=round(y*heigth)
#     #         print("Go to "+str(x)+' '+str(y))
#     #         self.raspberry_camera.moveServoMotorToCoordinates(origins=(width,heigth),destination_coordinates=(x,y),speed=0.009)
#     #         self.tracked_coordinates=(None,None)

#     def drawCenterLinesTarget(self,frame):
#         heigth,width=frame.shape[:2]
        
#         line_size=75
#         thickness=7
#         if self.raspberry_camera:
#             line_size=75 if self.raspberry_camera.zoom==1 else 75/self.raspberry_camera.zoom
#             thickness=7 if self.raspberry_camera.zoom==1 else int(7/self.raspberry_camera.zoom)

#         cv2.line(frame, (  int(width/2 - line_size),int(heigth/2)), (  int(width/2 + line_size),int(heigth/2)),  ( 0,255, 0),  thickness, cv2.LINE_AA)
#         cv2.line(frame, (  int(width/2),int(heigth/2-line_size)), (  int(width/2 ),int(heigth/2+line_size)),  ( 0,255, 0),  thickness, cv2.LINE_AA)
       
#     def addTrackingAndDetectionTimeAndFPS(self,img,detection_time,tracking_time,tracking_fps):
#         width=img.shape[1]
#         cv2.rectangle(img,(int(width-195),10),(int(width-10),85),color=(240,240,240),thickness=-1)
#         cv2.putText(img, f'FPS: {round(tracking_fps,2)}', (int(width-190),30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (25,25,250), 2)
#         cv2.putText(img, f'Det. time: {round(detection_time*1000)}ms', (int(width-190),52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (250,25,25), 2)
#         cv2.putText(img, f'Tra. time: {round(tracking_time*1000)}ms', (int(width-190),73), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (250,25,25), 2)
 
#     def returnSelectedTrackedObject(self,coordinates):
#         self.mouse_tracked_coordinates=coordinates
#         if self.mouse_tracked_coordinates:
#             x,y = self.mouse_tracked_coordinates
#             heigth,width=self.frame_size[:2]
#             print(heigth,width)
#             print( x,y)
#             x=round(x*width)
#             y=round(y*heigth)
            
#             for track in self.tracker.tracks:
#                 if ( track.is_confirmed() and track.time_since_update <=1):
#                     # FONCTION ERROR
#                     (l,t,r,b) = track.to_tlbr()
#                     if y>=t and y<=b and x>=l and x<=r:
#                         self.mouse_tracked_coordinates=None
#                         print("AN OBJECT TO TRACK IS FOUNDED : " +str(track.track_id)+" - "+str(track.to_tlwh()) )
#                         return track
#             self.mouse_tracked_coordinates=None
#             print("NO OBJECT FOUNDED")
#             return None
        
#         # if self.raspberry_camera.tracked_object and self.raspberry_camera.tracked_object.is_confirmed():
#         #     print("CONTINUE TRACKING  - 30 TTL : "+str(self.raspberry_camera.tracked_object.time_since_update )) 
#         #     return self.raspberry_camera.tracked_object





















#         # def  calculateFeatures (self,frame,bbox):
#         # x,y,w,h=bbox
        

#         # # print(bbox)
#         # sub_frame=frame[y:y+h, x:x+w] 
#         # # hist1 = cv2.calcHist([sub_frame],[0],None,[256],[0,256])
#         # # feature =np.array(hist1)
      
#         # # feature =np.array([])
#         # # feature =np.array(list(bbox))
        
#         # gray = cv2.cvtColor(sub_frame, cv2.COLOR_BGR2GRAY)

#         # # Detect keypoints in the image
#         # # keypoints = self.feature_detector.detect(gray)
#         # kp, descriptors = self.feature_detector.detectAndCompute(gray, None)
       
#         # cv2.BFMatcher()
#         # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

#         # # Match the descriptors
#         # matches = bf.match(descriptors, descriptors)
#         # matches = sorted(matches, key=lambda x: x.distance)

#         # if(matches and len(matches)>0):
#         #     print(matches[0].distance)
#         # # if(descriptors):
#         # # print(list(np.array(descriptors).flat))
#         # # print(descriptors.tolist())
#         # # if(descriptors):
#         # #     print(len(descriptors))

#         # # feature =np.array(np.array(descriptors).flatten())
#         # # feature =np.array(descriptors).flatten()
#         # feature =np.array([])

#         # white_rect = np.zeros(sub_frame.shape, dtype=np.uint8) 
#         # white_rect[:, :, 1] = 255
#         # res = cv2.addWeighted(sub_frame, 0.5, white_rect, 0.5, 1.0)
 
#         # res = cv2.drawKeypoints(gray, kp, outImage=None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#         # frame[y:y+h, x:x+w]=res
#         # # feature=np.array( )
#         # return feature