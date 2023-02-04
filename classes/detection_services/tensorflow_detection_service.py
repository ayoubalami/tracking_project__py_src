from http import server
import cv2,time,os,numpy as np
from classes.detection_services.detection_service import IDetectionService
from symbol import return_stmt
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import get_file 

class TensorflowDetectionService(IDetectionService):

    np.random.seed(123)
    model=None
    
    def clean_memory(self):
        print("CALL DESTRUCTER FROM TensorflowDetectionService")
        if self.model:
            del self.model
        tf.keras.backend.clear_session()
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
                        {'date':'20200711','name': 'ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8' },
                        {'date':'20200711','name': 'ssd_mobilenet_v2_320x320_coco17_tpu-8' },
                        {'date':'20200711','name': 'efficientdet_d0_coco17_tpu-32' },
                        {'date':'20200711','name': 'efficientdet_d1_coco17_tpu-32' },
                        {'date':'20200711','name': 'efficientdet_d3_coco17_tpu-32' },
                        {'date':'20200711','name': 'efficientdet_d7_coco17_tpu-32' },
                        {'date':'20200711','name': 'centernet_resnet50_v2_512x512_kpts_coco17_tpu-8' },
                        {'date':'20200713','name': 'centernet_hg104_512x512_coco17_tpu-8' },
                        {'date':'20200711','name': 'faster_rcnn_resnet50_v1_640x640_coco17_tpu-8' },
                        {'date':'20200711','name': 'faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8' }
                        # {'date':'20200711','name': 'faster_rcnn_resnet101_v1_640x640_coco17_tpu-8' },
                        # {'date':'20200711','name': 'faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8' },
                    ]
        self.init_object_detection_models_list()
    
        # https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
        # https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API

        # self.load_model()
        
    def service_name(self):
        return "Tensorflow detection service V 1.0"

    def load_model(self,model=None):
        self.load_or_download_model_tensorflow(model=model)
        # init a random tensor to speed up start button
        inputTensor=[[[0,0,0],[0,0,0]]]
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8) 
        inputTensor = inputTensor[tf.newaxis,...]
        self.model(inputTensor)
        
    def get_selected_model(self):
        return self.selected_model

    def load_or_download_model_tensorflow(self,model=None):
        self.selected_model = next(m for m in self.detection_method_list_with_url if m["name"] == model)
        self.modelURL= self.selected_model['url']
        print("===> selected modelURL")
        print(self.modelURL)
        fileName = os.path.basename(self.modelURL)     
        self.modelName = fileName[:fileName.index('.')]
        self.cacheDir = os.path.join("","models","tensorflow_models", self.modelName)
        os.makedirs(self.cacheDir, exist_ok=True)
        get_file(fname=fileName,origin=self.modelURL, cache_dir=self.cacheDir, cache_subdir="checkpoints",  extract=True)
        print("===> clear_session")
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
    

    def detect_objects(self, frame,threshold= 0.5,nms_threshold=0.5):
        inputTensor = cv2.cvtColor( frame.copy(), cv2.COLOR_BGR2RGB ) 
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8) 
        inputTensor = inputTensor[tf.newaxis,...]

        start_time= time.perf_counter()
        detections = self.model(inputTensor)
        inference_time=round(time.perf_counter()-start_time,3)

        bboxs = detections['detection_boxes'][0].numpy()
        classIndexes = detections['detection_classes'][0].numpy().astype(np.int32) 
        classScores = detections['detection_scores'][0].numpy()
        imH, imW, imC = frame.shape
    
        #     NON MAX SUPRESSION
        bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=150, 
            iou_threshold=nms_threshold, score_threshold=threshold)
 
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

        fps=1/ round( time.perf_counter()-start_time,3)
        self.addFrameFps(frame,fps)
        return frame,inference_time

    def init_object_detection_models_list(self):
        url_template = "http://download.tensorflow.org/models/object_detection/tf2/{date}/{name}.tar.gz"
        self.detection_method_list_with_url=[ {'date':model['date'] , 'name' :model['name'] , 'url': url_template.format(date = model['date'] ,name=model['name'])}  for model in self.detection_method_list ]

    def get_object_detection_models(self):
        return self.detection_method_list 
      
