# https://colab.research.google.com/drive/1V-F3erKkPun-vNn28BoOc6ENKmfo8kDh?usp=sharing#scrollTo=PlAqR7PJmvTL
from csv import writer
from http import server
import cv2,time,os,numpy as np
from classes.detection_services.detection_service import IDetectionService
from utils_lib.utils_functions import runcmd
import torch
class Yolov5DetectionService(IDetectionService):

    np.random.seed(123)
    model=None
    default_model_input_size=640

    def clean_memory(self):
        print("CALL DESTRUCTER FROM Yolov5DetectionService")
        if self.model:
            del self.model
        # tf.keras.backend.clear_session()
        # del self
   
    def __init__(self):
        self.classFile ="coco.names" 
        self.modelName=None
        # self.cacheDir=None
        self.classesList=None
        self.detection_method_list    =   [ 
                        {'name': 'yolov5n'   },
                        # {'name': 'yolov5n6'   },
                        {'name': 'yolov5s'  },
                        # {'name': 'yolov5s6'  },
                        {'name': 'yolov5m'  },
                        {'name': 'yolov5l'  },
                        {'name': 'yolov5x'  }                        
                       ]
        self.init_object_detection_models_list()
    
    def service_name(self):
        return "torch hub YOLOV5 detection service V 1.0"

    def load_model(self,model=None):
        self.selected_model = next(m for m in self.detection_method_list_with_url if m["name"] == model)
        self.modelName= self.selected_model['name']
        self.model = torch.hub.load('ultralytics/yolov5', self.modelName )         
        self.readClasses()
    
    def init_object_detection_models_list(self):
        self.detection_method_list_with_url=self.detection_method_list

    def get_object_detection_models(self):
        return self.detection_method_list 
      
    def detect_objects(self, frame,boxes_plotting=True ):
        start_time = time.perf_counter()
        if self.network_input_size!=None and self.network_input_size != self.default_model_input_size:
            self.default_model_input_size=self.network_input_size
            print("UPDATE YOLO V5 NETWORK INPUT SIZE ... "+str(self.default_model_input_size))
        # frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        labels, cord , inference_time = self.score_frame(frame)
        if boxes_plotting:
            frame , _ = self.plot_boxes((labels, cord ), frame,threshold=self.threshold,nms_threshold=self.nms_threshold,boxes_plotting=True)
            fps = 1 / np.round(time.perf_counter()-start_time,3)
            self.addFrameFps(frame,fps)
            return frame,inference_time
        else:
            return self.plot_boxes((labels, cord ), frame,threshold=self.threshold,nms_threshold=self.nms_threshold,boxes_plotting=False)
    
    def score_frame(self, frame):
        # self.model.to(self.device)
        frame = [frame]        
        start_time = time.perf_counter()
        results = self.model(frame,size=self.default_model_input_size)
        end_time = time.perf_counter()
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord , np.round(end_time - start_time, 4)

    def plot_boxes(self, results, frame,threshold,nms_threshold,boxes_plotting=True):
        boxes=[]
        confidences=[]
        classes_ids=[]

        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if float(row[4]) >= threshold:
                confidences.append(float(row[4]))
                classes_ids.append(int(labels[i]))
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                box = np.array([x1,y1,x2, y2])
                boxes.append(box) 
                
        indices = cv2.dnn.NMSBoxes(boxes,confidences,score_threshold=threshold,nms_threshold=nms_threshold)

        raw_detection_data=[]

        for i in indices:
            x1, y1, x2, y2 = boxes[i]
            label = self.classesList[classes_ids[i]]
            color = self.colors_list[classes_ids[i]]
            conf = confidences[i]
            if (boxes_plotting):                
                displayText = '{}: {:.2f}'.format(label, conf) 
                cv2.rectangle(frame,(x1,y1),(x2,y2),color=color,thickness=2)
                cv2.putText(frame, displayText, (x1,y1-2),cv2.FONT_HERSHEY_PLAIN, 1.5,color,2)
            else:
                raw_detection_data.append(([x1, y1, x2-x1, y2-y1],conf,label))

        return frame,raw_detection_data
      

 