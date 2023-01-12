# https://colab.research.google.com/drive/1V-F3erKkPun-vNn28BoOc6ENKmfo8kDh?usp=sharing#scrollTo=PlAqR7PJmvTL
from csv import writer
import cv2,time,os,numpy as np
from classes.detection_service import IDetectionService
#from ultralytics import YOLO

class Yolov8DetectionService(IDetectionService):

    np.random.seed(123)
    model=None
    
    def clean_memory(self):
        print("CALL DESTRUCTER FROM Yolov8DetectionService")
        if self.model:
            del self.model
        # tf.keras.backend.clear_session()
        # del self
   
    def __init__(self):
        self.perf = []
        self.classAllowed=[]
        self.colorList=[]
        # self.classFile ="models/coco.names" 
        self.classFile ="coco.names" 
        self.modelName=None
        # self.cacheDir=None
        self.classesList=None
        self.colorList=None
        self.classAllowed=[0,1,2,3,5,6,7]  # detected only person, car , bicycle ... 
        # self.classAllowed=range(0, 80)
        self.detection_method_list    =   [ 
                        {'name': 'yolov8n'   },
                        {'name': 'yolov5n6'   },
                        {'name': 'yolov5s'  },
                        {'name': 'yolov5s6'  },
                        {'name': 'yolov5m'  },
                        {'name': 'yolov5l'  },
                        {'name': 'yolov5x'  }                        
                       ]
        self.init_object_detection_models_list()
    
    def service_name(self):
        return "torch hub YOLOV8 detection service V 1.0"

    def load_model(self,model=None):
        self.selected_model = next(m for m in self.detection_method_list_with_url if m["name"] == model)
        self.modelName= self.selected_model['name']
        self.model = YOLO(self.modelName+".pt" )         
        self.readClasses()
    
    def get_selected_model(self):
        return self.selected_model

    def readClasses(self): 
        with open(self.classFile, 'r') as f:
            self.classesList = f.read().splitlines()
        #   delete all class except person and vehiccule 
        # self.classesList=self.classesList[0:8]
        # self.classesList.pop(4)
        print(self.classesList)
        # set Color of box for each object
        self.colorList =  [[23.82390253, 213.55385765, 104.61775798],
            [168.73771775, 240.51614241,  62.50830085],
            [  3.35575698,   6.15784347, 240.89335156],
            [235.76073062, 119.16921962,  95.65283276],
            [138.42940829, 219.02379358, 166.29923782],
            [ 59.40987365, 197.51795215,  34.32644182],
            [ 42.21779254, 156.23398212,  60.88976857]]
    
    def init_object_detection_models_list(self):
        self.detection_method_list_with_url=self.detection_method_list

    def get_object_detection_models(self):
        return self.detection_method_list 
      
    def detect_objects(self, frame,threshold= 0.5,nms_threshold= 0.5):
        start_time = time.perf_counter()
        img=frame.copy()
        self.model.predict_cli(source=frame,return_outputs=False,save=False)  
        #    
        return frame, 0
        # labels, cord , inference_time = self.score_frame(img)
        # img = self.plot_boxes((labels, cord ), img,threshold=threshold)
        # fps = 1 / np.round(time.perf_counter()-start_time,3)
        # self.addFrameFps(img,fps)
        return img,inference_time
        
    def score_frame(self, frame):
        # self.model.to(self.device)
        # frame = [frame]        
        start_time = time.perf_counter()
        # self.model.predict(source=frame,return_outputs=False,save=False)  
        # #    
        # return frame, 0
        # results = self.model(frame,size=640)
        # print(results)
        # print(len(results))

        return
        end_time = time.perf_counter()
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord , np.round(end_time - start_time, 4)

    def plot_boxes(self, results, frame,threshold):
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
                
        indices = cv2.dnn.NMSBoxes(boxes,confidences,score_threshold=threshold,nms_threshold=0.5)
   
        for i in indices:
            x1, y1, x2, y2 = boxes[i]
            classColor = (236,106,240)
            label = self.classesList[classes_ids[i]]
            if (classes_ids[i] in self.classAllowed)==True:
                classColor = self.colorList[self.classAllowed.index(classes_ids[i])]
            conf = confidences[i]
            displayText = '{}: {:.2f}'.format(label, conf) 
            cv2.rectangle(frame,(x1,y1),(x2,y2),color=classColor,thickness=2)
            cv2.putText(frame, displayText, (x1,y1-2),cv2.FONT_HERSHEY_PLAIN, 1.5,classColor,2)
        return frame

    
    
