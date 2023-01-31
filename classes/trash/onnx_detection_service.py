# # https://colab.research.google.com/drive/1V-F3erKkPun-vNn28BoOc6ENKmfo8kDh?usp=sharing#scrollTo=PlAqR7PJmvTL
# from http import server
# import cv2,time,os,numpy as np
# from classes.detection_service import IDetectionService

# class OnnxDetectionService(IDetectionService):

#     np.random.seed(123)
    
#     model=None
    
#     def clean_memory(self):
#         print("CALL DESTRUCTER FROM PytorchDetectionService")
#         if self.model:
#             del self.model
#         # tf.keras.backend.clear_session()
#         # del self
   
#     def __init__(self):
#         self.perf = []
#         self.classAllowed=[]
#         self.colorList=[]
#         # self.classFile ="models/coco.names" 
#         self.classFile ="coco.names" 
#         self.modelName=None
#         # self.cacheDir=None
#         self.classesList=None
#         self.colorList=None
#         # self.classAllowed=[0,1,2,3,5,6,7]  # detected only person, car , bicycle ... 
#         self.classAllowed=range(0, 80)  # detected only person, car , bicycle ... 

#         self.detection_method_list    =   [ 
#                         {'name': 'nanodet-plus-m-1.5x_320'  },
#                         {'name': 'nanodet-plus-m_320'  },
#                         {'name': 'yolov5n'  },
#                         {'name': 'yolov5s'  },
#                         {'name': 'yolov5m'  },
#                         {'name': 'yolov5l'  },
#                         {'name': 'yolov5x'  },
#                         {'name': 'yolov6n','url':'https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6n.onnx'  },
#                         {'name': 'yolov6t','url':'https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6t.onnx'  },
#                         {'name': 'yolov6s','url':'https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6s.onnx'  },
#                         {'name': 'yolov6m','url':'https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6m.onnx'  },
#                         {'name': 'yolov6l','url':'https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6l.onnx'  },
                       
#                         # {'name': 'object_detection_nanodet_2022nov'},
#                        ]

#         self.init_object_detection_models_list()
    
#     def service_name(self):
#         return "Pytorch detection service V 1.0"

#     def load_model(self,model=None):
#         self.selected_model = next(m for m in self.detection_method_list_with_url if m["name"] == model)
#         self.modelName= self.selected_model['name']
#         modelgPath=os.path.join("","models","opencv_onnx_models",self.modelName+".onnx")
#         self.model = cv2.dnn.readNetFromONNX(modelgPath)       
#         self.readClasses()

#     def get_selected_model(self):
#         return self.selected_model

#     def readClasses(self): 
#         with open(self.classFile, 'r') as f:
#             self.classesList = f.read().splitlines()
#         #   delete all class except person and vehiccule 
#         self.classesList=self.classesList[0:8]
#         self.classesList.pop(4)
#         # print(self.classesList)
#         # set Color of box for each object
#         self.colorList =  [[23.82390253, 213.55385765, 104.61775798],
#             [168.73771775, 240.51614241,  62.50830085],
#             [  3.35575698,   6.15784347, 240.89335156],
#             [235.76073062, 119.16921962,  95.65283276],
#             [138.42940829, 219.02379358, 166.29923782],
#             [ 59.40987365, 197.51795215,  34.32644182],
#             [ 42.21779254, 156.23398212,  60.88976857]]
    
        

#     def unletterbox(self,bbox, original_image_shape, letterbox_scale):
#         ret = bbox.copy()

#         h, w = original_image_shape
#         top, left, newh, neww = letterbox_scale

#         if h == w:
#             ratio = h / newh
#             ret = ret * ratio
#             return ret

#         ratioh, ratiow = h / newh, w / neww
#         ret[0] = max((ret[0] - left) * ratiow, 0)
#         ret[1] = max((ret[1] - top) * ratioh, 0)
#         ret[2] = min((ret[2] - left) * ratiow, w)
#         ret[3] = min((ret[3] - top) * ratioh, h)
#         return ret.astype(np.int32)


        
#     def letterbox(self,srcimg, target_size=(416, 416)):
#         img = srcimg.copy()

#         top, left, newh, neww = 0, 0, target_size[0], target_size[1]
#         if img.shape[0] != img.shape[1]:
#             hw_scale = img.shape[0] / img.shape[1]
#             if hw_scale > 1:
#                 newh, neww = target_size[0], int(target_size[1] / hw_scale)
#                 img = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)
#                 left = int((target_size[1] - neww) * 0.5)
#                 img = cv2.copyMakeBorder(img, 0, 0, left, target_size[1] - neww - left, cv2.BORDER_CONSTANT, value=0)  # add border
#             else:
#                 newh, neww = int(target_size[0] * hw_scale), target_size[1]
#                 img = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)
#                 top = int((target_size[0] - newh) * 0.5)
#                 img = cv2.copyMakeBorder(img, top, target_size[0] - newh - top, 0, 0, cv2.BORDER_CONSTANT, value=0)
#         else:
#             img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

#         letterbox_scale = [top, left, newh, neww]
#         return img, letterbox_scale
        

#     def post_process(self, preds):
#         cls_scores, bbox_preds = preds[::2], preds[1::2]
#         rescale = False
#         scale_factor = 1
#         bboxes_mlvl = []
#         scores_mlvl = []
#         for stride, cls_score, bbox_pred, anchors in zip(self.strides, cls_scores, bbox_preds, self.anchors_mlvl):
#             if cls_score.ndim==3:
#                 cls_score = cls_score.squeeze(axis=0)
#             if bbox_pred.ndim==3:
#                 bbox_pred = bbox_pred.squeeze(axis=0)

#             x_exp = np.exp(bbox_pred.reshape(-1, self.reg_max + 1))
#             x_sum = np.sum(x_exp, axis=1, keepdims=True)
#             bbox_pred = x_exp / x_sum
#             bbox_pred = np.dot(bbox_pred, self.project).reshape(-1,4)
#             bbox_pred *= stride

#             nms_pre = 1000
#             if nms_pre > 0 and cls_score.shape[0] > nms_pre:
#                 max_scores = cls_score.max(axis=1)
#                 topk_inds = max_scores.argsort()[::-1][0:nms_pre]
#                 anchors = anchors[topk_inds, :]
#                 bbox_pred = bbox_pred[topk_inds, :]
#                 cls_score = cls_score[topk_inds, :]

#             points = anchors
#             distance = bbox_pred
#             max_shape=self.image_shape
#             x1 = points[:, 0] - distance[:, 0]
#             y1 = points[:, 1] - distance[:, 1]
#             x2 = points[:, 0] + distance[:, 2]
#             y2 = points[:, 1] + distance[:, 3]

#             if max_shape is not None:
#                 x1 = np.clip(x1, 0, max_shape[1])
#                 y1 = np.clip(y1, 0, max_shape[0])
#                 x2 = np.clip(x2, 0, max_shape[1])
#                 y2 = np.clip(y2, 0, max_shape[0])

#             #bboxes = np.stack([x1, y1, x2, y2], axis=-1)
#             bboxes = np.column_stack([x1, y1, x2, y2])
#             bboxes_mlvl.append(bboxes)
#             scores_mlvl.append(cls_score)

#         bboxes_mlvl = np.concatenate(bboxes_mlvl, axis=0)
#         if rescale:
#             bboxes_mlvl /= scale_factor
#         scores_mlvl = np.concatenate(scores_mlvl, axis=0)
#         bboxes_wh = bboxes_mlvl.copy()
#         bboxes_wh[:, 2:4] = bboxes_wh[:, 2:4] - bboxes_wh[:, 0:2]
#         classIds = np.argmax(scores_mlvl, axis=1)
#         confidences = np.max(scores_mlvl, axis=1)

#         indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), self.prob_threshold, self.iou_threshold)

#         if len(indices)>0:
#             det_bboxes = bboxes_mlvl[indices]
#             det_conf = confidences[indices]
#             det_classid = classIds[indices]

#             return np.concatenate([det_bboxes, det_conf.reshape(-1, 1), det_classid.reshape(-1, 1)], axis=1)
#         else:
#             return np.array([])

#     def detect_objects(self, frame,threshold= 0.5,nms_threshold= 0.5):
        
#         ret=frame.copy()
#         ret = cv2.resize(ret, (416 ,416 ))
#         ret = cv2.cvtColor(ret, cv2.COLOR_BGR2RGB)
#         ret, letterbox_scale = self.letterbox(ret)

#         ret = ret.astype(np.float32)
#         self.mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
#         self.std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)

#         ret = (ret - self.mean) / self.std
#         blob = cv2.dnn.blobFromImage(ret)

#         self.model.setInput(blob)
#         outs = self.model.forward()
#         preds = self.post_process(outs)

#         # draw bboxes and labels
#         for pred in preds:
#             bbox = pred[:4]
#             conf = pred[-2]
#             classid = pred[-1].astype(np.int32)

#             # bbox
#             xmin, ymin, xmax, ymax = self.unletterbox(bbox, ret.shape[:2], letterbox_scale)
#             cv2.rectangle(ret, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=2)

#             # label
#             label = "{:s}: {:.2f}".format("classes[classid]", conf)

#         return ret , 0



#     # def detect_objects(self, frame,threshold= 0.5,nms_threshold= 0.5):
        
#     #     if  self.model ==None:
#     #         return None,0

#     #     img=frame.copy()
#     #     blob_size=640
#     #     # img = cv2.resize(img, (1280 ,1280 ))
#     #     blob = cv2.dnn.blobFromImage(img,scalefactor= 1/255,size=(blob_size ,blob_size ),mean=[0,0,0],swapRB= True, crop= False)
#     #     self.model.setInput(blob)

#     #     t1= time.time()
#     #     detections = self.model.forward()[0]
#     #     inference_time=time.time()-t1
        
#     #     classes_ids = []
#     #     confidences = []
#     #     boxes = []
        
#     #     rows = detections.shape[0]

#     #     img_width, img_height = img.shape[1], img.shape[0]
#     #     x_scale = img_width/blob_size
#     #     y_scale = img_height/blob_size
#     #     # print(detections)
#     #     # print(range(rows))
#     #     # input("Please enter to get confidence:\n")    
#     #     for i in range(rows):
#     #         row = detections[i]
#     #         box_confidence = float(row[4]) 
#     #         if box_confidence > threshold:
#     #             classes_confidences = row[5:]
#     #             ind = np.argmax(classes_confidences)
#     #             object_confidence= classes_confidences[ind]
#     #             if object_confidence > threshold:
#     #                 classes_ids.append(ind)
#     #                 # confidence= classes_score[ind]*confidence
#     #                 confidences.append(object_confidence)
#     #                 cx, cy, w, h = row[:4]
#     #                 x1 = int((cx- w/2)*x_scale)
#     #                 y1 = int((cy-h/2)*y_scale)
#     #                 width = int(w * x_scale)
#     #                 height = int(h * y_scale)
#     #                 box = np.array([x1,y1,width,height])
#     #                 boxes.append(box)              
#     #     indices = cv2.dnn.NMSBoxes(boxes,confidences,score_threshold=threshold,nms_threshold=nms_threshold)
#     #     for i in indices:
#     #         x1,y1,w,h = boxes[i]
#     #         if (classes_ids[i] in self.classAllowed)==False:
#     #             continue
#     #         # label = self.classesList[classes_ids[i]]
#     #         label = self.classesList[self.classAllowed.index(classes_ids[i])]
#     #         classColor = self.colorList[self.classAllowed.index(classes_ids[i])]
#     #         conf = confidences[i]
                            
#     #         displayText = '{}: {:.2f}'.format(label, conf) 

#     #         cv2.rectangle(img,(x1,y1),(x1+w,y1+h),color=classColor,thickness=2)
#     #         cv2.putText(img, displayText, (x1,y1-2),cv2.FONT_HERSHEY_PLAIN, 1.5,classColor,2)
                    
#     #     return img , inference_time




#     # def get_object_detection_models(self):
#     #     # url_template = "http://download.tensorflow.org/models/object_detection/tf2/{date}/{name}.tar.gz"
#     #     # url_list=[ {'date':model['date'] , 'name' :model['name'] , 'url': url_template.format(date = model['date'] ,name=model['name'])}  for model in list ]
#     #     # url_list=[  {'name' :model['name'] , 'url': model['name']}  for model in list ]
#     #     url_list=[  {'name' :model['name'] }  for model in list ]

#     #     return url_list

#     def init_object_detection_models_list(self):
#         # to add network config url dynamicly
#         # url_template_cfg = "https://github.com/AlexeyAB/darknet/tree/master/cfg/{name}.cfg" 
#         # self.detection_method_list_with_url=[ { 'name' :model['name'] ,'url_weights' :model['url_weights'] , 'url_cfg': url_template_cfg.format( name=model['name']) }  for model in self.detection_method_list ]
#         self.detection_method_list_with_url=self.detection_method_list

#     def get_object_detection_models(self):
#         return self.detection_method_list 
      
