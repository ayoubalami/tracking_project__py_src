# https://colab.research.google.com/drive/1V-F3erKkPun-vNn28BoOc6ENKmfo8kDh?usp=sharing#scrollTo=PlAqR7PJmvTL
# ls /usr/local/lib/python3.9/site-packages/
from csv import writer
from http import server
import cv2,time,os,numpy as np
from classes.detection_services.detection_service import IDetectionService
from utils_lib.utils_functions import runcmd
import torch
from utils_lib.nms import non_max_suppression

class Yolov6DetectionService(IDetectionService):

    np.random.seed(123)
    model=None
    default_model_input_size=640

    def clean_memory(self):
        print("CALL DESTRUCTER FROM Yolov6DetectionService")
        if self.model:
            del self.model
        # tf.keras.backend.clear_session()
        # del self
   
    def __init__(self):
        self.classFile ="coco.names" 
        self.modelName=None
        # self.cacheDir=None
        self.classesList=None
        self.detection_method_list    =   [ 
                        {'name': 'yolov6n'   },
                        {'name': 'yolov6s'  },
                        {'name': 'yolov6m'  },
                        {'name': 'yolov6l'  },
                        # {'name': 'yolov6lite_s'  },
                        # {'name': 'yolov6lite_m'  },
                        # {'name': 'yolov6lite_l'  },
                        # {'name': 'yolov6lite_s'  },            
                        # {'name': 'yolov6s_mbla'  },    
                        # {'name': 'yolov6m_mbla'  },    
                        # {'name': 'yolov6l_mbla'  },    
                       ]
        self.init_object_detection_models_list()
    
    def service_name(self):
        return "torch hub YOLOV6 detection service V 1.0"

    def load_model(self,model=None):
        self.selected_model = next(m for m in self.detection_method_list_with_url if m["name"] == model)
        self.modelName= self.selected_model['name']
        self.model = torch.hub.load('/root/YOLOv6' , self.modelName , source='local' ,force_reload=True)  
        # self.model = torch.hub.load('/root/YOLOv6' ,'custom',  path =self.modelName+'.pt' , source='local' ,force_reload=True)  

        self.readClasses()
    # https://github.com/meituan/YOLOv6/releases/tag/0.4.0
    def init_object_detection_models_list(self):
        self.detection_method_list_with_url=self.detection_method_list

    def get_object_detection_models(self):
        return self.detection_method_list 
      
    def detect_objects(self, frame,boxes_plotting=True ):
        start_time = time.perf_counter()
        if self.network_input_size!=None and self.network_input_size != self.default_model_input_size:
            self.default_model_input_size=self.network_input_size
            print("UPDATE YOLO V5 NETWORK INPUT SIZE ... "+str(self.default_model_input_size))
        # frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prediction , inference_time = self.score_frame(frame)
        if boxes_plotting:
            frame , _ = self.plot_boxes(prediction, frame,boxes_plotting=True)
            fps = 1 / np.round(time.perf_counter()-start_time,3)
            self.addFrameFps(frame,fps)
            return frame,inference_time
        else:
            return self.plot_boxes(prediction, frame,boxes_plotting=False)
    
    def score_frame(self, frame):
        
        self.device=torch.device('cpu')
        start_time = time.perf_counter()
        self.img_size=self.default_model_input_size
        img, img_src = self.process_image(frame, self.img_size, 32)
        img = img.to(self.device)
        if len(img.shape) == 3:
            img = img[None]

        self.model.device = self.device
        # self.model.iou_thres = self.nms_threshold
        # print(f" self.model.conf_thres { self.model.conf_thres} , self.model.iou_thres :{self.model.iou_thres } ")
        prediction = self.model.forward(img, img_src.shape)
        return prediction , np.round(time.perf_counter() - start_time, 4)

    def plot_boxes(self, prediction, frame,boxes_plotting=True):
        raw_detection_data=[]
        if len( prediction['boxes'])==0:
            return frame,[]
        boxes= prediction['boxes'].int().numpy()
        scores = prediction['scores'].detach().numpy()
        classes= prediction['labels'].int().numpy()
        indices = cv2.dnn.NMSBoxes(boxes,scores,score_threshold=self.threshold,nms_threshold=self.nms_threshold)
        for i in range(len(indices)):
            x1, y1, x2, y2=boxes[i]
            if (boxes_plotting):                
                displayText = '{}: {:.2f}'.format(classes[i], scores[i]) 
                cv2.rectangle(frame,(x1,y1),(x2,y2),color=self.colors_list[classes[i]],thickness=2)
                cv2.putText(frame, displayText, (x1,y1-2),cv2.FONT_HERSHEY_PLAIN, 1.5,self.colors_list[classes[i]],2)
            else:
                raw_detection_data.append(([x1, y1, x2-x1, y2-y1],scores[i],classes[i]))
        return frame,raw_detection_data
      
    def letterbox(self,im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        '''Resize and pad image while meeting stride-multiple constraints.'''
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        elif isinstance(new_shape, list) and len(new_shape) == 1:
            new_shape = (new_shape[0], new_shape[0])

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (left, top)


    def process_image(self,img_src, img_size, stride):
        '''Preprocess image before inference.'''
        image = self.letterbox(img_src, img_size, stride=stride)[0]
        image = image.transpose((2, 0, 1)) # HWC to CHW
        image = torch.from_numpy(np.ascontiguousarray(image))
        image = image.float()
        image /= 255
        return image, img_src

    @staticmethod       
    def rescale(ori_shape, boxes, target_shape):
        '''Rescale the output to the original image shape'''
        ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
        padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

        boxes[:, [0, 2]] -= padding[0]
        boxes[:, [1, 3]] -= padding[1]
        boxes[:, :4] /= ratio

        boxes[:, 0].clamp_(0, target_shape[1])  # x1
        boxes[:, 1].clamp_(0, target_shape[0])  # y1
        boxes[:, 2].clamp_(0, target_shape[1])  # x2
        boxes[:, 3].clamp_(0, target_shape[0])  # y2

        return boxes
