# https://colab.research.google.com/drive/1V-F3erKkPun-vNn28BoOc6ENKmfo8kDh?usp=sharing#scrollTo=PlAqR7PJmvTL
from csv import writer
from http import server
import cv2,time,os,numpy as np
from classes.detection_service import IDetectionService
from utils_lib.utils_functions import runcmd
import torch
class OnnxDetectionService(IDetectionService):

    np.random.seed(123)
    model=None
    
    def clean_memory(self):
        print("CALL DESTRUCTER FROM OnnxDetectionService")
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
                        # {'name': 'nanodet-plus-m-1.5x_320'  },
                        # {'name': 'nanodet-plus-m_320'  },
                        # {'name': 'yolov5n_err2'  },

                        {'name': 'yolov5n' , 'url':'https://github.com/ayoubalami/flask_python/releases/download/v0.1.0/yolov5n.onnx'  },
                        {'name': 'yolov5s' , 'url':'https://github.com/ayoubalami/flask_python/releases/download/v0.1.0/yolov5s.onnx' },
                        {'name': 'yolov5m' , 'url':'https://github.com/ayoubalami/flask_python/releases/download/v0.1.0/yolov5m.onnx' },
                        {'name': 'yolov5l' , 'url':'https://github.com/ayoubalami/flask_python/releases/download/v0.1.0/yolov5l.onnx' },
                        {'name': 'yolov5x' , 'url':'https://github.com/ayoubalami/flask_python/releases/download/v0.1.0/yolov5x.onnx' },
                        {'name': 'yolov6n','url':'https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6n.onnx'  },
                        {'name': 'yolov6t','url':'https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6t.onnx'  },
                        {'name': 'yolov6s','url':'https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6s.onnx'  },
                        {'name': 'yolov6m','url':'https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6m.onnx'  },
                        {'name': 'yolov6l','url':'https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6l.onnx'  },
                        {'name': 'yolov8n','url':'_'},
                       
                       ]

        self.init_object_detection_models_list()
    
    def service_name(self):
        return "ONNX detection service V 1.0"

    def download_model_if_not_exists(self):
        print("===> download_model_if_not_exists  ")
        remote_onnx_file= self.selected_model['url']
        self.modelName =self.selected_model['name']
        cacheDir = os.path.join("","models","opencv_onnx_models")
        print("downloading",remote_onnx_file," ..")
        if not os.path.exists(   os.path.join(cacheDir,  self.modelName+'.onnx'   )):
            print("===> download_model onnx")
            os.makedirs(cacheDir, exist_ok=True)
            runcmd("wget -P " + cacheDir + "   " + remote_onnx_file, verbose = True)   
        else:
            print("===> model onnx already exist ")
        self.modelPath=os.path.join("","models","opencv_onnx_models",self.modelName+".onnx")

    def load_model(self,model=None):
        self.selected_model = next(m for m in self.detection_method_list_with_url if m["name"] == model)
        self.modelName= self.selected_model['name']
        self.download_model_if_not_exists()
        self.model = cv2.dnn.readNetFromONNX(self.modelPath)    
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
    
    def detect_objects(self, frame,threshold= 0.5,nms_threshold= 0.5,boxes_plotting=True):
        
        if  self.model ==None:
            return None,0

        img=frame.copy()
        blob_size=640

        blob = cv2.dnn.blobFromImage(img,scalefactor= 1/255,size=(blob_size ,blob_size ),mean=[0,0,0],swapRB= True, crop= False)
        self.model.setInput(blob)
        start_time= time.perf_counter()
        detections = self.model.forward()[0]
        inference_time=round(time.perf_counter()-start_time,3)
        classes_ids = []
        confidences = []
        boxes = []
        rows = detections.shape[0]
        img_width, img_height = img.shape[1], img.shape[0]
        x_scale = img_width/blob_size
        y_scale = img_height/blob_size
        
        for i in range(rows):
            row = detections[i]
            box_confidence = float(row[4]) 
            if box_confidence > threshold:
                classes_confidences = row[5:]
                ind = np.argmax(classes_confidences)
                object_confidence= classes_confidences[ind]
                if object_confidence > threshold:
                    classes_ids.append(ind)
                    # confidence= classes_score[ind]*confidence
                    confidences.append(object_confidence)
                    cx, cy, w, h = row[:4]
                    x1 = int((cx- w/2)*x_scale)
                    y1 = int((cy-h/2)*y_scale)
                    width = int(w * x_scale)
                    height = int(h * y_scale)
                    box = np.array([x1,y1,width,height])
                    boxes.append(box)              
        indices = cv2.dnn.NMSBoxes(boxes,confidences,score_threshold=threshold,nms_threshold=nms_threshold)
        raw_detection_data=[]

        for i in indices:
            x1,y1,w,h = boxes[i]
            label = self.classesList[classes_ids[i]]
            classColor = (236,106,240)
            if (classes_ids[i] in self.classAllowed)==True:
                label = self.classesList[classes_ids[i]]
                classColor = self.colorList[self.classAllowed.index(classes_ids[i])]
            conf = confidences[i]
            displayText = '{}: {:.2f}'.format(label, conf) 

            if boxes_plotting :
                cv2.rectangle(img,(x1,y1),(x1+w,y1+h),color=classColor,thickness=2)
                cv2.putText(img, displayText, (x1,y1-2),cv2.FONT_HERSHEY_PLAIN, 1.5,classColor,2)
            else:
                raw_detection_data.append(([x1, y1, w, h],conf,label))

        if boxes_plotting :
            fps= 1 /round(time.perf_counter()-start_time,3)
            self.addFrameFps(img,fps)
            return img,inference_time
        else:
            return img,raw_detection_data

    def init_object_detection_models_list(self):
        self.detection_method_list_with_url=self.detection_method_list

    def get_object_detection_models(self):
        return self.detection_method_list 
       