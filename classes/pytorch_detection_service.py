# https://colab.research.google.com/drive/1V-F3erKkPun-vNn28BoOc6ENKmfo8kDh?usp=sharing#scrollTo=PlAqR7PJmvTL
from http import server
import cv2,time,os,numpy as np
from classes.detection_service import IDetectionService

class PytorchDetectionService(IDetectionService):

    np.random.seed(123)
    model=None
    
    def clean_memory(self):
        print("CALL DESTRUCTER FROM PytorchDetectionService")
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

        self.detection_method_list    =   [ 
                        {'name': 'nanodet-plus-m-1.5x_320'  },
                        {'name': 'nanodet-plus-m_320'  },
                        {'name': 'yolov5n'  },
                        {'name': 'yolov5n_err'  },
                        {'name': 'yolov5s'  },
                        {'name': 'yolov5m'  },
                        {'name': 'yolov5l'  },
                        {'name': 'yolov5x'  },
                        {'name': 'yolov6n','url':'https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6n.onnx'  },
                        {'name': 'yolov6t','url':'https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6t.onnx'  },
                        {'name': 'yolov6s','url':'https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6s.onnx'  },
                        {'name': 'yolov6m','url':'https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6m.onnx'  },
                        {'name': 'yolov6l','url':'https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6l.onnx'  },
                       
                        # {'name': 'object_detection_nanodet_2022nov'},
                       ]

        self.init_object_detection_models_list()
    
    def service_name(self):
        return "Pytorch detection service V 1.0"

    def load_model(self,model=None):
        self.selected_model = next(m for m in self.detection_method_list_with_url if m["name"] == model)
        self.modelName= self.selected_model['name']
        modelgPath=os.path.join("","models","opencv_onnx_models",self.modelName+".onnx")
        self.model = cv2.dnn.readNetFromONNX(modelgPath)       
        self.readClasses()

    def get_selected_model(self):
        return self.selected_model

    def readClasses(self): 
        with open(self.classFile, 'r') as f:
            self.classesList = f.read().splitlines()
        #   delete all class except person and vehiccule 
        self.classesList=self.classesList[0:8]
        self.classesList.pop(4)
        # print(self.classesList)
        # set Color of box for each object
        self.colorList =  [[23.82390253, 213.55385765, 104.61775798],
            [168.73771775, 240.51614241,  62.50830085],
            [  3.35575698,   6.15784347, 240.89335156],
            [235.76073062, 119.16921962,  95.65283276],
            [138.42940829, 219.02379358, 166.29923782],
            [ 59.40987365, 197.51795215,  34.32644182],
            [ 42.21779254, 156.23398212,  60.88976857]]
    
          

    def detect_objects(self, frame,threshold= 0.5,nms_threshold= 0.5):
        
        if  self.model ==None:
            return None,0

        img=frame.copy()
        # img = cv2.resize(img, (int(1280/4) ,int(720/4) ))
        blob_size=640
        # img = cv2.resize(img, (1280 ,1280 ))
        blob = cv2.dnn.blobFromImage(img,scalefactor= 1/255,size=(blob_size ,blob_size ),mean=[0,0,0],swapRB= True, crop= False)
        self.model.setInput(blob)
        t1= time.time()
        detections = self.model.forward()[0]
        inference_time=time.time()-t1
        
        classes_ids = []
        confidences = []
        boxes = []
        
        rows = detections.shape[0]

        img_width, img_height = img.shape[1], img.shape[0]
        x_scale = img_width/blob_size
        y_scale = img_height/blob_size
        # print(detections)
        # print(range(rows))
        # input("Please enter to get confidence:\n")    
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
        for i in indices:
            x1,y1,w,h = boxes[i]
            if (classes_ids[i] in self.classAllowed)==False:
                continue
            # label = self.classesList[classes_ids[i]]
            label = self.classesList[self.classAllowed.index(classes_ids[i])]
            classColor = self.colorList[self.classAllowed.index(classes_ids[i])]
            conf = confidences[i]
                            
            displayText = '{}: {:.2f}'.format(label, conf) 

            cv2.rectangle(img,(x1,y1),(x1+w,y1+h),color=classColor,thickness=2)
            cv2.putText(img, displayText, (x1,y1-2),cv2.FONT_HERSHEY_PLAIN, 1.5,classColor,2)
                
        # print(round(inference_time,2))
            
        return img , inference_time


    def init_object_detection_models_list(self):
        self.detection_method_list_with_url=self.detection_method_list

    def get_object_detection_models(self):
        return self.detection_method_list 
      
