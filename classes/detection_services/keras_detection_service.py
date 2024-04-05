from http import server
import cv2,time,os,numpy as np
from classes.detection_services.detection_service import IDetectionService
from symbol import return_stmt
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import get_file 

class KerasDetectionService(IDetectionService):

    np.random.seed(123)
    model=None
    default_model_input_size=196

    def clean_memory(self):
        print("CALL DESTRUCTER FROM KerasDetectionService")
        if self.model:
            del self.model
        tf.keras.backend.clear_session()
        # del self
   
    def __init__(self):
        self.perf = []
        self.classAllowed=[]
        self.colorList=[]
        # self.classFile ="det_models/coco.names" 
        self.classFile ="coco.names" 
        self.modelName=None
        self.cacheDir=None
        self.classesList=None
        self.colorList=None
        self.classAllowed=[0,1,2,3,5,6,7]  # detected only person, car , bicycle ... 
        self.selected_model=None
        self.detection_method_list    =   [ 
                        {'folder':'yolov5s_saved_model_416','input_size':416, 'name': 'yolov5s_416' },
                        {'folder':'yolov5n_saved_model_320','input_size':320,'name': 'yolov5n_320' },
                        {'folder':'yolov5n_saved_model_192','input_size':192,'name': 'yolov5n_192' },
                           ]
        self.init_object_detection_models_list()
    
        # https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
        # https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API

        # self.load_model()
        
    def service_name(self):
        return "Keras detection service V 1.0"

    def load_model(self,model=None):
        tf.keras.backend.clear_session()
        self.selected_model = next(m for m in self.detection_method_list if m["name"] == "yolov5n_192")  
        print("===> init load_model" +self.selected_model['name'])
        self.modelName = os.path.join("","det_models","keras_models", self.selected_model['folder'])
        self.model = tf.keras.models.load_model(self.modelName)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("Model of KERAS " + self.modelName + " loaded successfully...")
        self.readClasses()

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
        self.classesList.insert(0,-1)
        self.colorList.insert(0,-1)
    
    def preprocess_image(self,frame):
       
         # image = tf.keras.preprocessing.image.load_img(image_src, target_size=(blob_size, blob_size))
        frame=cv2.resize(frame, (self.selected_model['input_size'],self.selected_model['input_size']))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the BGR image to RGB (OpenCV loads images in BGR format)
        image = tf.keras.preprocessing.image.img_to_array(frame)
        image = tf.expand_dims(image, axis=0)
        image=image/255
        return image

    def detect_objects(self, frame):
        bounding_boxes=[]
        confidence_scores=[]
        classes=[]
        anchors=[]
        H , W=frame.shape[:2]

        # print("detect_objects-->")
        preprocessed_image=self.preprocess_image(frame)

        start_time= time.perf_counter()
        predictions = self.model.predict(preprocessed_image)
        inference_time=round(time.perf_counter()-start_time,3)

        for i,element in enumerate(predictions[0][0]):
            if element[4]>self.threshold:
                max_conf = np.max(element[5:])
                # max_index = np.argmax(element[5:])
                max_class = np.argmax(element[5:])
                if max_conf>self.threshold and max_class == 2:
                    box=element[:4]
                    x1,y1,width,height=box
                    x1=x1-width/2
                    y1=y1-height/2
                    # i+=1
                    # print(x1*W,y1*H,width*W,height*H)
                    x, y, width, height= np.array([x1*W,y1*H,width*W,height*H]).astype(int)
                    # cv2.rectangle(imageS, ((int)(x), (int)(y)), ((int)(x+width), (int)(y+height)), (200,0,0), 2)
                    bounding_boxes.append(((int)(x), (int)(y), (int)(x+width), (int)(y+height)))
                    confidence_scores.append(max_conf)
                    classes.append(max_class)
                    anchors.append(i)

        indices = cv2.dnn.NMSBoxes(bounding_boxes, confidence_scores,  score_threshold=self.threshold,nms_threshold=self.nms_threshold)
        nms_filtered_boxes = [ (index,anchors[i],bounding_boxes[i]) for index,i in enumerate(indices)]
        for index,anchor_index,(x, y, x2, y2) in nms_filtered_boxes:
            cv2.rectangle(frame, ((int)(x), (int)(y)), ((int)(x2), (int)(y2)), (200,0,200), 2)

        fps=1/ round( time.perf_counter()-start_time,3)
        self.addFrameFps(frame,fps)
        return frame,inference_time

    # def detect_objects(self, frame):
    #     inputTensor = cv2.cvtColor( frame.copy(), cv2.COLOR_BGR2RGB ) 
    #     inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8) 
    #     inputTensor = inputTensor[tf.newaxis,...]

    #     start_time= time.perf_counter()
    #     detections = self.model(inputTensor)
    #     inference_time=round(time.perf_counter()-start_time,3)

    #     bboxs = detections['detection_boxes'][0].numpy()
    #     classIndexes = detections['detection_classes'][0].numpy().astype(np.int32) 
    #     classScores = detections['detection_scores'][0].numpy()
    #     imH, imW, imC = frame.shape
    
    #     #     NON MAX SUPRESSION
    #     bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=150, 
    #         iou_threshold=self.nms_threshold, score_threshold=self.threshold)
 
    #     if len(bboxIdx) != 0: 
    #         for i in bboxIdx: 
    #             bbox = tuple(bboxs[i].tolist())
    #             classConfidence =np.round(100*classScores[i])
    #             classIndex = classIndexes[i]

    #             if (classIndex not in self.classAllowed ):
    #                 continue

    #             classLabelText=self.classesList[self.classAllowed.index(classIndex)]
    #             classColor = self.colorList[self.classAllowed.index(classIndex)]

    #             # classColor = colorList[classIndex]
    #             displayText = '{}: {}'.format(classLabelText, classConfidence) 
    #             ymin, xmin, ymax, xmax = bbox
    #             xmin, xmax, ymin, ymax = (xmin * imW, xmax * imW, ymin * imH, ymax * imH) 
    #             xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

    #             cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=classColor, thickness=2) 
    #             cv2.putText(frame, displayText, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

    #     fps=1/ round( time.perf_counter()-start_time,3)
    #     self.addFrameFps(frame,fps)
    #     return frame,inference_time

    def init_object_detection_models_list(self):
        pass
    