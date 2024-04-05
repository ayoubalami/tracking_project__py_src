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
        self.classFile ="coco.names" 
        self.modelName=None
        self.cacheDir=None
        self.classesList=None
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
 
    def service_name(self):
        return "Tensorflow Lite detection service V 1.0"

    def load_model(self,model=None):

        self.selected_model = next(m for m in self.detection_method_list_with_url if m["name"] == model)
        self.modelName= self.selected_model['name']+'.tflite'
        self.const_network_input_size= self.selected_model['size']
        self.cacheDir = os.path.join("","det_models","tensorflow_tflite_models", self.modelName)
        self.model = tf.lite.Interpreter(model_path=self.cacheDir)
        self.model.allocate_tensors()
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()
        print("Model " + self.modelName + " loaded successfully...")
        self.readClasses()
 
    def detect_objects(self, frame,boxes_plotting=True):         
        image_data = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (self.const_network_input_size,self.const_network_input_size ))
        images_list = np.asarray([image_data]).astype(np.uint8)  
        start_time= time.perf_counter()  
        self.model.set_tensor(self.input_details[0]['index'], images_list)
        self.model.invoke()
        inference_time=round(time.perf_counter()-start_time,3)

        output_dict = {
        'num_detections': int(self.model.get_tensor(self.output_details[3]["index"])),
        'detection_classes': self.model.get_tensor(self.output_details[1]["index"]).astype(np.uint8),
        'detection_boxes' : self.model.get_tensor(self.output_details[0]["index"]),
        'detection_scores' : self.model.get_tensor(self.output_details[2]["index"])
        }

        heigth,width=frame.shape[:2]
        class_idx=list(output_dict['detection_classes'][0])
        bboxes=list(output_dict['detection_boxes'][0])
        confidences=list(output_dict['detection_scores'][0])
        # total_detections=output_dict['num_detections']
 
        raw_detection_data=[]
        allowed_condidats=self.keepSelectedClassesOnly(bboxes,confidences,class_idx,self.threshold)
        if not allowed_condidats :
            if boxes_plotting :
                fps = 1 / np.round(time.perf_counter()-start_time,3)
                self.addFrameFps(frame,fps)
                return frame,inference_time
            else:
                return frame,raw_detection_data

        bboxes,confidences,class_idx=list(zip(*allowed_condidats))        
        nms_bbox_idx = tf.image.non_max_suppression(bboxes, confidences, max_output_size=150, 
            iou_threshold=self.nms_threshold, score_threshold=self.threshold)

        for i in nms_bbox_idx:
            class_id=class_idx[i]
            bbox=bboxes[i]
            confidence=round(confidences[i],3)
            if confidence>self.threshold:
                ymin, xmin, ymax, xmax = bbox
                xmin, xmax, ymin, ymax = (xmin * width, xmax * width, ymin * heigth, ymax * heigth) 
                xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
                label_text=self.classesList[class_id]
                color = self.colors_list[class_id]
                if boxes_plotting :
                    displayText = str(label_text)+' '+ str(confidence)
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=color, thickness=2) 
                    cv2.putText(frame, displayText, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
                else:
                    raw_detection_data.append(([int(xmin), int(ymin),int(xmax-xmin), int(ymax-ymin)],confidence,label_text))
        if boxes_plotting :
            fps = 1 / np.round(time.perf_counter()-start_time,3)
            self.addFrameFps(frame,fps)
            return frame,inference_time
        else:
            return frame,raw_detection_data


        