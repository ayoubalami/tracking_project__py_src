# https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
from http import server
import cv2,time,os,numpy as np
from classes.detection_services.detection_service import IDetectionService
from utils_lib.utils_functions import runcmd
import random
class OpencvDetectionService(IDetectionService):

    np.random.seed(123)
    model=None
    default_model_input_size=416
    show_all_classes=True

    def clean_memory(self):
        print("CALL DESTRUCTER FROM OpencvDetectionService")
        if self.model:
            del self.model
        # tf.keras.backend.clear_session()
        # del self
   
    def __init__(self):
        self.perf = []
        self.classAllowed=[]
        self.colorList=[]
        self.classFile ="coco.names" 
        self.modelName=None
        self.classesList=None
        self.colorList=None
        self.readClasses()
        # self.classAllowed=[0,1,2,3,5,6,7]  # detected only person, car , bicycle ... 
        # self.classAllowed=range(0, 80)
        self.allowed_classes=[1,2,3]
        self.selected_model=None
        self.detection_method_list    =   [ 
                        # {'name': 'yolov2' , 'url_cfg': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov2.cfg' , 'url_weights' :'https://pjreddie.com/media/files/yolov2.weights' },
                        {'type':'yolo','name': 'yolov3' , 'url_cfg': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3.cfg' , 'url_weights' :'https://pjreddie.com/media/files/yolov3.weights'},
                        {'type':'yolo','name': 'yolov4' , 'url_cfg': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg' ,'url_weights' :'https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights' },
                        {'type':'yolo','name': 'yolov7' , 'url_cfg': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov7.cfg' ,'url_weights' :'https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov7.weights'  },
                        {'type':'yolo','name': 'yolov3-tiny' , 'url_cfg': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3-tiny.cfg' ,'url_weights' :'https://pjreddie.com/media/files/yolov3-tiny.weights'},
                        {'type':'yolo','name': 'yolov4-tiny'  , 'url_cfg': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg' ,'url_weights' :'https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights'},
                        {'type':'yolo','name': 'yolov7-tiny' , 'url_cfg': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov7-tiny.cfg' ,'url_weights' :'https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov7-tiny.weights'},
                        {'type':'ssd','name':"ssd_mobilenet_v3_large_coco_2020_01_14","url_cfg":"https://raw.githubusercontent.com/haroonshakeel/real_time_object_detection_cpu/main/model_data/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt","url_weights":"https://github.com/haroonshakeel/real_time_object_detection_cpu/raw/main/model_data/frozen_inference_graph.pb"}
                        ]
        self.init_object_detection_models_list()
               
    def service_name(self):
        return "opencv detection service V 1.0"

    def download_model_if_not_exists(self):
        model_url_cfg= self.selected_model['url_cfg']
        model_url_weights= self.selected_model['url_weights']
        self.modelName =self.selected_model['name']
        cacheDir = os.path.join("","models","opencv_models", self.modelName)
        print( self.selected_model)
        if self.selected_model['type']=='yolo':
            if not os.path.exists(   os.path.join(cacheDir,  self.modelName+'.cfg'   )):
                print("===> download_model cfg")
                os.makedirs(cacheDir, exist_ok=True)
                runcmd("wget -P " + cacheDir + "   " + model_url_cfg, verbose = True)
            else:
                print("===> model cfg already exist ")

            if not os.path.exists(   os.path.join(cacheDir,  self.modelName+'.weights'    )):
                print("===> download_model weights")
                os.makedirs(cacheDir, exist_ok=True)
                runcmd("wget -P " + cacheDir + "   " + model_url_weights, verbose = True)
            else:
                print("===> model weights already exist ")
            self.configPath=os.path.join("","models","opencv_models",self.modelName,self.modelName+".cfg")
            self.weightPath=os.path.join("","models","opencv_models",self.modelName,self.modelName+".weights")
        else : 
            if self.selected_model['type']=='ssd':
                if not os.path.exists(   os.path.join(cacheDir, self.modelName+'.pbtxt'   )):
                    print("===> download_model cfg")
                    os.makedirs(cacheDir, exist_ok=True)
                    runcmd("wget -P " + cacheDir + "   " + model_url_cfg, verbose = True)
                else:
                    print("===> model cfg already exist ")

                if not os.path.exists(   os.path.join(cacheDir,  'frozen_inference_graph.pb'    )):
                    print("===> download_model weights")
                    os.makedirs(cacheDir, exist_ok=True)
                    runcmd("wget -P " + cacheDir + "   " + model_url_weights, verbose = True)
                else:
                    print("===> model weights already exist ")
            self.configPath=os.path.join("","models","opencv_models",self.modelName,self.modelName+".pbtxt")
            self.weightPath=os.path.join("","models","opencv_models",self.modelName,"frozen_inference_graph.pb")


    def load_model(self,model=None):

        self.selected_model = next(m for m in self.detection_method_list_with_url if m["name"] == model)  
        self.modelName= self.selected_model['name']
        print("===> selected modelName : " +self.modelName )
        self.download_model_if_not_exists()
        net = cv2.dnn.readNet(self.configPath,self.weightPath)
        # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        self.model=cv2.dnn_DetectionModel(net)
        if self.selected_model['type']=='ssd':
            self.model.setInputParams(size=(512, 512),mean=(127.5, 127.5, 127.5), scale=1.0/127.5, swapRB=True)
            self.classesList .insert(0,"__")
            self.colorList .insert(0,(0,0,0))

        elif self.selected_model['type']=='yolo':
            self.model.setInputParams(size=(self.default_model_input_size, self.default_model_input_size), scale=1/255, swapRB=True)

        print("Model " + self.modelName + " loaded successfully...")
      
    def get_selected_model(self):
        return self.selected_model

    def readClasses(self): 
        with open(self.classFile, 'r') as f:
            self.classesList = f.read().splitlines()
        #   delete all class except person and vehiccule 
        # self.classesList=self.classesList[0:8]
        # self.classesList.pop(4)
        print(self.classesList)
        # set static Colors of 8 first classes
        self.colorList =    [[23, 213, 104],
                            [168, 240,  62],
                            [  3,   6, 240],
                            [235, 119,  95],
                            [138, 219, 166],
                            [ 59, 197,  34],
                            [ 42, 156,  60],
                            [ 142, 56,  250]]
        # add random colors to remaining colors
        more_random_color=[[random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)] for i in range(73)] 
        self.colorList+=more_random_color

    def detect_objects(self, frame,threshold:float,nms_threshold:float,boxes_plotting=True):
       
        start_time= time.perf_counter()
        if self.network_input_size!=None and self.network_input_size != self.default_model_input_size:
            self.default_model_input_size=self.network_input_size
            print("UPDATE YOLO NETWORK INPUT SIZE ... ")
            self.model.setInputParams(size=(self.default_model_input_size, self.default_model_input_size), scale=1/255, swapRB=True)
             
        classIdx,confidences,bboxs= self.model.detect(frame,confThreshold=threshold)

        inference_time=np.round(time.perf_counter()-start_time,3)
        bboxs=list(bboxs)
        confidences=list(confidences)
        classIdx=list(classIdx)
        raw_detection_data=[]
        allowed_condidats=self.keepSelectedClass(bboxs,confidences,classIdx)
        if not allowed_condidats :
            if boxes_plotting :
                fps = 1 / np.round(time.perf_counter()-start_time,3)
                self.addFrameFps(frame,fps)
                return frame,inference_time
            else:
                return frame,raw_detection_data
        else :
            bboxs,confidences,classIdx=list(zip(*allowed_condidats))
            bboxIdx=cv2.dnn.NMSBoxes(bboxs,confidences,score_threshold=threshold,nms_threshold=nms_threshold)
            if len(bboxIdx) !=0 :        
                for i in range (0,len(bboxIdx)):
                    bbox=bboxs[np.squeeze(bboxIdx[i])]
                    classConfidence = confidences[bboxIdx[i]]
                    classLabelID=np.squeeze(classIdx[bboxIdx[i]])
                    classLabel = self.classesList[classLabelID]
                    classColor  = self.colorList[classLabelID]
                    displayText = '{}: {:.2f}'.format(classLabel, classConfidence)
                    x,y,w,h=bbox
                    if boxes_plotting :
                        cv2.rectangle(frame,(x,y),(x+w,y+h),color=classColor,thickness=2)
                        cv2.putText(frame, displayText, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)
                    else:
                        raw_detection_data.append(([x, y, w, h],classConfidence,classLabel))
            if boxes_plotting :
                fps = 1 / np.round(time.perf_counter()-start_time,3)
                self.addFrameFps(frame,fps)
                return frame,inference_time
            else:
                return frame,raw_detection_data


    def init_object_detection_models_list(self):
        self.detection_method_list_with_url=self.detection_method_list

    def get_object_detection_models(self):
        return self.detection_method_list 
      

    def keepSelectedClass(self,bboxs,confidences,classIdx):
      
        condidats=zip(bboxs,confidences,classIdx)
        condidats=list(condidats)
         # allowed_condidats = [condidat for index, condidat in enumerate(list(condidats)) if condidat[2] in self.allowed_classes]
        allowed_condidats = [condidat for index, condidat in enumerate(condidats) if condidat[2] in self.allowed_classes]
 
        if len(allowed_condidats)>0:
            return allowed_condidats

        return None
        # if len(condidats)>0:
        #     print("condidat[2]")
        #     print(list(condidats)[0][2])
        
        # result = list(filter(lambda allowed_class: ( allowed_class in classIdx ), self.allowed_classes )) 
        # print(result)
        # return bboxs,confidences,classIdx