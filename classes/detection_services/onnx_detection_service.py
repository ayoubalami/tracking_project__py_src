# https://colab.research.google.com/drive/1V-F3erKkPun-vNn28BoOc6ENKmfo8kDh?usp=sharing#scrollTo=PlAqR7PJmvTL
from csv import writer
from http import server
import cv2,time,os,numpy as np
from classes.detection_services.detection_service import IDetectionService
from utils_lib.utils_functions import runcmd
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
        self.classFile ="coco.names" 
        self.modelName=None
        self.classesList=None
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
    
    def detect_objects(self, frame,boxes_plotting=True):
        
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
            color = self.colors_list[classes_ids[i]]
            conf = confidences[i]
            displayText = '{}: {:.2f}'.format(label, conf) 
            if boxes_plotting :
                cv2.rectangle(img,(x1,y1),(x1+w,y1+h),color=color,thickness=2)
                cv2.putText(img, displayText, (x1,y1-2),cv2.FONT_HERSHEY_PLAIN, 1.5,color,2)
            else:
                raw_detection_data.append(([x1, y1, w, h],conf,label))

        if boxes_plotting :
            fps= 1 /round(time.perf_counter()-start_time,3)
            self.addFrameFps(img,fps)
            return img,inference_time
        else:
            return img,raw_detection_data


       