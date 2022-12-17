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
        self.selected_model=None
        self.detection_method_list    =   [ 
                        {'name': 'yolov5n'  },
                        {'name': 'yolov5s'  },
                        {'name': 'yolov5m'  },
                        {'name': 'yolov5l'  },
                        {'name': 'yolov5x'  }
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
      
    

    # def detect_objects(self, frame,threshold= 0.5):
    #     # return frame
    #     return self. detect_objects_non_max_suppression(frame,threshold)


    def detect_objects(self, frame,threshold= 0.5,nms_threshold= 0.5):

        img=frame.copy()
        # img = cv2.resize(img, (1280 ,1280 ))

        blob = cv2.dnn.blobFromImage(img,scalefactor= 1/255,size=(640 ,640 ),mean=[0,0,0],swapRB= True, crop= False)
        self.model.setInput(blob)
        detections = self.model.forward()[0]
        # [0]
        
        classes_ids = []
        confidences = []
        boxes = []
        rows = detections.shape[0]

        img_width, img_height = img.shape[1], img.shape[0]
        x_scale = img_width/640
        y_scale = img_height/640

        for i in range(rows):
            row = detections[i]
            confidence = row[4]
            if confidence > threshold:
                classes_score = row[5:]
                ind = np.argmax(classes_score)
                if classes_score[ind] > threshold:
                    classes_ids.append(ind)
                    confidences.append(confidence)
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
                    
        return img


    # def get_object_detection_models(self):
    #     # url_template = "http://download.tensorflow.org/models/object_detection/tf2/{date}/{name}.tar.gz"
    #     # url_list=[ {'date':model['date'] , 'name' :model['name'] , 'url': url_template.format(date = model['date'] ,name=model['name'])}  for model in list ]
    #     # url_list=[  {'name' :model['name'] , 'url': model['name']}  for model in list ]
    #     url_list=[  {'name' :model['name'] }  for model in list ]

    #     return url_list

    def init_object_detection_models_list(self):
        # to add network config url dynamicly
        # url_template_cfg = "https://github.com/AlexeyAB/darknet/tree/master/cfg/{name}.cfg" 
        # self.detection_method_list_with_url=[ { 'name' :model['name'] ,'url_weights' :model['url_weights'] , 'url_cfg': url_template_cfg.format( name=model['name']) }  for model in self.detection_method_list ]
        self.detection_method_list_with_url=self.detection_method_list

    def get_object_detection_models(self):
        return self.detection_method_list 
      
