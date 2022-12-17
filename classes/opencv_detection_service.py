
from http import server
import cv2,time,os,numpy as np
from classes.detection_service import IDetectionService
from symbol import return_stmt
import subprocess


class OpencvDetectionService(IDetectionService):

    np.random.seed(123)
   
    model=None
    
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
        # self.classFile ="models/coco.names" 
        self.classFile ="coco.names" 
        self.modelName=None
        # self.cacheDir=None
        self.classesList=None
        self.colorList=None
        self.classAllowed=[0,1,2,3,5,6,7]  # detected only person, car , bicycle ... 
        self.selected_model=None
        self.detection_method_list    =   [ 
                        {'name': 'yolov2' , 'url_cfg': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov2.cfg' , 'url_weights' :'https://pjreddie.com/media/files/yolov2.weights' },
                        {'name': 'yolov3' , 'url_cfg': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3.cfg' , 'url_weights' :'https://pjreddie.com/media/files/yolov3.weights'},
                        {'name': 'yolov4' , 'url_cfg': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg' ,'url_weights' :'https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights' },
                        {'name': 'yolov7' , 'url_cfg': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov7.cfg' ,'url_weights' :'https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov7.weights'  },
                        {'name': 'yolov3-tiny' , 'url_cfg': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3-tiny.cfg' ,'url_weights' :'https://pjreddie.com/media/files/yolov3-tiny.weights'},
                        {'name': 'yolov4-tiny'  , 'url_cfg': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg' ,'url_weights' :'https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights'},
                        {'name': 'yolov7-tiny' , 'url_cfg': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov7-tiny.cfg' ,'url_weights' :'https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov7-tiny.weights'}
                        ]

        self.init_object_detection_models_list()
    
        # https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
       
        # self.load_model()
        
    def service_name(self):
        return "opencv detection service V 1.0"

    def runcmd(self,cmd, verbose = False, *args, **kwargs):

        process = subprocess.Popen(
            cmd,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text = True,
            shell = True
        )
        std_out, std_err = process.communicate()
        if verbose:
            print(std_out.strip(), std_err)
        pass

    def download_model_if_not_exists(self):
        print("===> download_model_if_not_exists  ")
        model_url_cfg= self.selected_model['url_cfg']
        model_url_weights= self.selected_model['url_weights']
 
        print( self.selected_model)
        # print( model_url_cfg)
        # print( model_url_weights)
        fileName = os.path.basename(model_url_cfg)     
        self.modelName = fileName[:fileName.index('.')]
        cacheDir = os.path.join("","models","opencv_models", self.modelName)

        # print(os.path.exists( os.path.join(cacheDir,  fileName    )))
        # print(  os.path.join(cacheDir,  fileName   ))
        print( self.selected_model)
        print( self.modelName+'.cfg')
        print( self.modelName+'.weights'  )

        if not os.path.exists(   os.path.join(cacheDir,  self.modelName+'.cfg'    )):
            print("===> download_model cfg")
            os.makedirs(cacheDir, exist_ok=True)
            self.runcmd("wget -P " + cacheDir + "   " + model_url_cfg, verbose = True)
        else:
            print("===> model cfg already exist ")

        if not os.path.exists(   os.path.join(cacheDir,  self.modelName+'.weights'    )):
            print("===> download_model weights")
            os.makedirs(cacheDir, exist_ok=True)
            self.runcmd("wget -P " + cacheDir + "   " + model_url_weights, verbose = True)
        else:
            print("===> model weights already exist ")
            

    def load_model(self,model=None):
        # self.load_or_download_model_tensorflow(model=model)
        # model="yolov4"
        self.selected_model = next(m for m in self.detection_method_list_with_url if m["name"] == model)
  
        self.modelName= self.selected_model['name']
        print("===> selected modelName")
        print(self.modelName)
        self.download_model_if_not_exists()

        configPath=os.path.join("","models","opencv_models",self.modelName,self.modelName+".cfg")
        modelPath=os.path.join("","models","opencv_models",self.modelName,self.modelName+".weights")
        
        net = cv2.dnn.readNet(modelPath,configPath)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

        self.model=cv2.dnn_DetectionModel(net)
        self.model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
        print("Model " + self.modelName + " loaded successfully...")
        self.readClasses()
        # self.selected_model= self.model


    def get_selected_model(self):
        return self.selected_model

         
        


    def readClasses(self): 
        with open(self.classFile, 'r') as f:
            self.classesList = f.read().splitlines()
        #   delete all class except person and vehiccule 
        self.classesList=self.classesList[0:8]
        self.classesList.pop(4)
        print(self.classesList)
        # set Color of box for each object
        self.colorList =  [[23.82390253, 213.55385765, 104.61775798],
            [168.73771775, 240.51614241,  62.50830085],
            [  3.35575698,   6.15784347, 240.89335156],
            [235.76073062, 119.16921962,  95.65283276],
            [138.42940829, 219.02379358, 166.29923782],
            [ 59.40987365, 197.51795215,  34.32644182],
            [ 42.21779254, 156.23398212,  60.88976857]]
      
     


    def detect_objects(self, frame,threshold:float,nms_threshold:float):
      
        frame=frame.copy()
        classLabelIDs,confidences,bboxs= self.model.detect(frame,confThreshold=threshold)
    
        bboxs=list(bboxs)
        confidences=list(np.array(confidences).reshape(1,-1)[0])
        confidences=list(map(float,confidences))
        bboxIdx=cv2.dnn.NMSBoxes(bboxs,confidences,score_threshold=threshold,nms_threshold=nms_threshold)
             
        if len(bboxIdx) !=0 :
            for i in range (0,len(bboxIdx)):
                bbox=bboxs[np.squeeze(bboxIdx[i])]
                classConfidence = confidences[bboxIdx[i]]
                classLabelID=np.squeeze(classLabelIDs[bboxIdx[i]])
        
                if (classLabelID in self.classAllowed)==False:
                    continue

                classLabel = self.classesList[self.classAllowed.index(classLabelID)]
                classColor = self.colorList[self.classAllowed.index(classLabelID)]
                displayText = '{}: {:.2f}'.format(classLabel, classConfidence) 
                
                x,y,w,h=bbox
                cv2.rectangle(frame,(x,y),(x+w,y+h),color=classColor,thickness=2)
                cv2.putText(frame, displayText, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

            return frame
            
        return frame



    def init_object_detection_models_list(self):
        self.detection_method_list_with_url=self.detection_method_list

    def get_object_detection_models(self):
        return self.detection_method_list 
      
