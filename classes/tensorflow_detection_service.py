
from http import server
import cv2,time,os,numpy as np
from classes.detection_service import IDetectionService
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import get_file 

class TensorflowDetectionService(IDetectionService):

    np.random.seed(123)
    # threshold = 0.5
   
    def __init__(self):
        self.perf = []
        self.classAllowed=[]
        self.colorList=[]
        # self.classFile ="models/coco.names" 
        self.classFile ="coco.names" 
        self.modelName=None
        self.cacheDir=None
        self.model=None
        self.classesList=None
        self.colorList=None
        self.classAllowed=[0,1,2,3,5,6,7]  # detected only person, car , bicycle ... 

        # https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
        self.modelURL= "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
        # self.modelURL= "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz"
        # self.modelURL= "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz"
        # self.modelURL="http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz"
        ### self.modelURL="http://download.tensorflow.org/models/object_detection/tf2/20210210/centernet_mobilenetv2fpn_512x512_coco17_od.tar.gz"
        # self.modelURL="http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v2_512x512_coco17_tpu-8.tar.gz"
        # self.modelURL="http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_640x640_coco17_tpu-8.tar.gz"
        self.load_model()

        
    def service_name(self):
        return "Tensorflow detection service V 1.0"

    def model_name(self):
        return self.modelName
        
    def load_model(self):
        self.load_or_download_model_tensorflow()
        # init a random tensor to speed up start button
        inputTensor=[[[0,0,0],[0,0,0]]]
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8) 
        inputTensor = inputTensor[tf.newaxis,...]
        self.model(inputTensor)
        # a = tf.add(inputTensor, 1)
        # print(a.numpy())

    def load_or_download_model_tensorflow(self):

        fileName = os.path.basename(self.modelURL)     
        self.modelName = fileName[:fileName.index('.')]
        self.cacheDir = os.path.join("","models","tensorflow_models", self.modelName)
        os.makedirs(self.cacheDir, exist_ok=True)
        get_file(fname=fileName,origin=self.modelURL, cache_dir=self.cacheDir, cache_subdir="checkpoints",  extract=True)
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.cacheDir, "checkpoints", self.modelName, "saved_model"))
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
        self.classesList.insert(0,-1)
        self.colorList.insert(0,-1)
    


    def detect_objects(self, frame,threshold= 0.5):
        # return frame
        return self. detect_objects_non_max_suppression(frame,threshold)
 

    def detect_objects_non_max_suppression(self, frame,threshold= 0.5):
        
        inputTensor = cv2.cvtColor( frame.copy(), cv2.COLOR_BGR2RGB ) 
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8) 
        inputTensor = inputTensor[tf.newaxis,...]
        detections = self.model(inputTensor)

        bboxs = detections['detection_boxes'][0].numpy()
        classIndexes = detections['detection_classes'][0].numpy().astype(np.int32) 
        classScores = detections['detection_scores'][0].numpy()
        imH, imW, imC = frame.shape
    
        #     NON MAX SUPRESSION
        bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=150, 
            iou_threshold=0.2, score_threshold=threshold)
 
        if len(bboxIdx) != 0: 
            for i in bboxIdx: 
                bbox = tuple(bboxs[i].tolist())
                classConfidence =np.round(100*classScores[i])

                classIndex = classIndexes[i]

                if (classIndex not in self.classAllowed ):
                    continue

                classLabelText=self.classesList[self.classAllowed.index(classIndex)]
                classColor = self.colorList[self.classAllowed.index(classIndex)]

                # classColor = colorList[classIndex]
                displayText = '{}: {}'.format(classLabelText, classConfidence) 
                ymin, xmin, ymax, xmax = bbox
                xmin, xmax, ymin, ymax = (xmin * imW, xmax * imW, ymin * imH, ymax * imH) 
                xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=classColor, thickness=2) 
                cv2.putText(frame, displayText, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

        return frame

 

