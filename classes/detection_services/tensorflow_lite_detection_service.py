from http import server
import cv2,time,os,numpy as np
from classes.detection_services.detection_service import IDetectionService
# from symbol import return_stmt
import tensorflow as tf
# from tensorflow.python.keras.utils.data_utils import get_file 
# import tflite_runtime.interpreter as tflite
# 
class TensorflowLiteDetectionService(IDetectionService):

    np.random.seed(123)
    model=None
     
    def clean_memory(self):
        print("CALL DESTRUCTER FROM TensorflowLiteDetectionService")
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
        self.cacheDir=None
        self.classesList=None
        self.colorList=None
        self.classAllowed=[0,1,2,3,5,6,7]  # detected only person, car , bicycle ... 
        self.selected_model=None
        self.detection_method_list    =   [ 
                        { 'size':320,'name': 'lite-model_efficientdet_lite0_detection_default_1' },
                        { 'size':640,'name': 'lite-model_efficientdet_lite4_detection_default_2' },
                        { 'size':300,'name': 'ssd_mobilenet_v1_1_metadata_1' },
                        
                        { 'size':320,'name': 'ssd_mobilenet_v3_large_coco_2020_01_14' },
                        { 'size':320,'name': 'ssd_mobilenet_v3_small_coco_2020_01_14' },
                        
                        # {'date':'','name': 'object_detection_mobile_object_localizer_v1_1_default_1' }
                    ]
        self.init_object_detection_models_list()
    
        # https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
        # https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API

        # self.load_model()
        
    def service_name(self):
        return "Tensorflow Lite detection service V 1.0"

    def load_model(self,model=None):
        self.load_or_download_model_tensorflow(model=model) 
        
    def get_selected_model(self):
        return self.selected_model

    def load_or_download_model_tensorflow(self,model=None):

        self.selected_model = next(m for m in self.detection_method_list_with_url if m["name"] == model)
        self.modelName= self.selected_model['name']+'.tflite'
        self.const_network_input_size= self.selected_model['size']
        self.cacheDir = os.path.join("","models","tensorflow_tflite_models", self.modelName)
        self.model = tf.lite.Interpreter(model_path=self.cacheDir)
        self.model.allocate_tensors()
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()
        print("Model " + self.modelName + " loaded successfully...")
        self.readClasses()

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
        # self.classesList.insert(0,-1)
        # self.colorList.insert(0,-1)
    

    def detect_objects(self, frame,threshold= 0.5,nms_threshold=0.5):         
        image_data = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (self.const_network_input_size,self.const_network_input_size ))
        images_list = np.asarray([image_data]).astype(np.uint8)  
        start_time= time.perf_counter()  
        self.model.set_tensor(self.input_details[0]['index'], images_list)
        self.model.invoke()
       
        inference_time=round(time.perf_counter()-start_time,3)
        fps=1/ round( time.perf_counter()-start_time,3)
        self.addFrameFps(frame,fps)
        
        output_dict = {
        'num_detections': int(self.model.get_tensor(self.output_details[3]["index"])),
        'detection_classes': self.model.get_tensor(self.output_details[1]["index"]).astype(np.uint8),
        'detection_boxes' : self.model.get_tensor(self.output_details[0]["index"]),
        'detection_scores' : self.model.get_tensor(self.output_details[2]["index"])
        }

        heigth,width=frame.shape[:2]
        classes=output_dict['detection_classes'][0]
        bboxes=output_dict['detection_boxes'][0]
        scores=output_dict['detection_scores'][0]
        total_detections=output_dict['num_detections']
        # print(bboxes)
        nmsBboxIdx = tf.image.non_max_suppression(bboxes, scores, max_output_size=150, 
            iou_threshold=nms_threshold, score_threshold=threshold)

        # for i in range(total_detections):
        for i in nmsBboxIdx:
            class_id=classes[i]
            bbox=bboxes[i]
            score=round(scores[i],2)
            if score>threshold:
                ymin, xmin, ymax, xmax = bbox
                xmin, xmax, ymin, ymax = (xmin * width, xmax * width, ymin * heigth, ymax * heigth) 
                xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)


                if (class_id not in self.classAllowed ):
                        continue

                classLabelText=self.classesList[self.classAllowed.index(class_id)]
                classColor = self.colorList[self.classAllowed.index(class_id)]

                # if score>.2:
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=classColor, thickness=2) 
                displayText = str(classLabelText)+' '+ str(score)
                cv2.putText(frame, displayText, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

        return frame,inference_time

    def init_object_detection_models_list(self):
        self.detection_method_list_with_url=[ {'size':model['size'] , 'name' :model['name'] }  for model in self.detection_method_list ]

    def get_object_detection_models(self):
        return self.detection_method_list 