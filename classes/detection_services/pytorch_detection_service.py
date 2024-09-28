# https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
from http import server
from classes.tracking_service.feature_extractor.FeatureExtractor import FeatureExtractor
import cv2,time,os,numpy as np
from classes.detection_services.detection_service import IDetectionService
from utils_lib.utils_functions import runcmd
import random
import torch
 
class PytorchDetectionService(IDetectionService):

    np.random.seed(123)
    model=None
    default_model_input_size=416
    show_all_classes=True
    use_pretrained_model=True
    max_anchors_number=10
    min_surface_area=1000
    threshold=.35

    def clean_memory(self):
        print("CALL DESTRUCTER FROM PytorchDetectionService")
        if self.model:
            del self.model
               
    def __init__(self,anchors_number):

        self.max_anchors_number=(int)(anchors_number)
        if self.use_pretrained_model:
            self.classFile ="coco.names" 
        else:
            self.classFile ="best_4.names" 

        self.modelName=None
        self.classesList=None
        self.readClasses()
        self.selected_model=None
        self.detection_method_list =   [ 
                        # {'name': 'yolov2' , 'url_cfg': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov2.cfg' , 'url_weights' :'https://pjreddie.com/media/files/yolov2.weights' },
                        {'file':'yolov5m.pt','name': 'yolov5m' },
                        {'file':'yolov5s.pt','name': 'yolov5s' },
                        {'file':'yolov5n.pt','name': 'yolov5n' }
                      ]
        self.init_object_detection_models_list()

               
    def service_name(self):
        return f"PyTorch detection service V 1.0 |  max_anchors_number : { self.max_anchors_number} "

    def download_model_if_not_exists(self):
        pass

    def load_model(self,model=None):
        self.selected_model = next(m for m in self.detection_method_list_with_url if m["name"] == model)  
        self.modelName= self.selected_model['name']
        print("===> selected modelName : " +self.modelName )

        # self.model = torch.hub.load("ultralytics/yolov5", self.selected_model['name'], pretrained=True)
        # ckpt = torch.load(self.selected_model['file'] )
        # csd = ckpt['model'].float().state_dict()
        # self.model.load_state_dict(csd, strict=False)
        # self.model.eval()  
        # self.model=None
        if self.use_pretrained_model:
            self.classFile ="coco.names" 
            self.modelPath=os.path.join("det_models","torch","pretrained",self.selected_model['file'])  
        else:
            self.classFile ="best_4.names"
            self.modelPath=os.path.join("det_models","torch","best_6.pt")  

        self.modelDirPath=os.path.join("det_models","torch")  
        self.model = torch.hub.load('/root/shared/yolov5', 'custom', path=self.modelPath, source='local')
        self.target_layer_names = ["model.model.model.23","model.model.model.20","model.model.model.17","model.model.model.24"]
        self.feature_extractor_model = FeatureExtractor(self.model, self.target_layer_names)
        print("Model " + self.modelName + " loaded successfully...")
      

    ##############################

    def preprocess_cv2(self,frame):
        resized_image = cv2.resize(frame, (self.default_model_input_size,self.default_model_input_size))
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        normalized_image = rgb_image / 255.0
        normalized_image = np.expand_dims(normalized_image.transpose(2, 0, 1), axis=0)
        input_tensor = torch.tensor(normalized_image, dtype=torch.float32)
        return input_tensor

    def pass_by_yolo(self,frame,class_num=[2],threshold=0.5):
        input_tensor=self.preprocess_cv2(frame)
        self.feature_extractor_model(input_tensor)
        layer_feature_large = self.feature_extractor_model.get_layer_output(self.target_layer_names[0]).permute(0,3,2,1)
        layer_feature_medium = self.feature_extractor_model.get_layer_output(self.target_layer_names[1]).permute(0,3,2,1)
        layer_feature_small = self.feature_extractor_model.get_layer_output(self.target_layer_names[2]).permute(0,3,2,1)
        layer_detection = self.feature_extractor_model.get_layer_output(self.target_layer_names[3])[0]
        return (layer_feature_small,layer_feature_medium,layer_feature_large),layer_detection
        # [1,52,52,128],[1,26,26,256],[1,13,13,512],[1,10647,85]

    def from_id_to_featureMap(self,id,features):
        net_input=self.default_model_input_size
        small_anchors=(net_input/8)*(net_input/8)*3
        medium_anchors=(net_input/16)*(net_input/16)*3
        large_anchors=(net_input/32)*(net_input/32)*3
        feature_id=-1 # 0 => small, 1=> medium, 2 => large
        if id < small_anchors:
            y=(id/3/(net_input/8))
            x=(id/3)%(net_input/8)
            feature_id=0
        elif id < medium_anchors+small_anchors and small_anchors <= id:
            id=id-small_anchors
            y=(id/3/(net_input/16))
            x=(id/3)%(net_input/16)
            feature_id=1
        elif id >= medium_anchors+small_anchors:
            id=id-(medium_anchors+small_anchors)
            y=(id/3/(net_input/32))
            x=(id/3)%(net_input/32)
            feature_id=2
        return features[feature_id][0][(int)(y)][(int)(x)]

    def extract_valide_objects(self,frame,class_num=[2],threshold=0.5,return_only_detections=False):
        class_nums = np.array(class_num)
        features_array,predictions_array=self.pass_by_yolo(frame,class_num,threshold)
        predictions_array=predictions_array[0]
        H , W=frame.shape[:2]
        x_ratio=self.network_input_size/W
        y_ratio=self.network_input_size/H
        
        ####### Class name prediction #######
        # condition = torch.zeros(predictions_array.shape[0], dtype=torch.bool)
        # for class_num in class_nums:
        #     condition |= torch.argmax(predictions_array[:, 5:], axis=1) == class_num
        # condition &
         
        # print(predictions_array[:, 1])
        # print(self.network_input_size)

        surface_box = (predictions_array[:, 2] * predictions_array[:, 3])/(y_ratio*x_ratio)
        surface_condition = (surface_box < 150000) & (surface_box > self.min_surface_area)
        exited_box_condition=((predictions_array[:, 1]+predictions_array[:, 3]/2)/y_ratio<H-5 ) & ((predictions_array[:, 1]+predictions_array[:, 3]/2)/y_ratio>150 ) 
 
        condition= (
                (torch.amax(predictions_array[:, 5:], axis=1) > threshold)
            &   (predictions_array[:, 4] > threshold)
            &   (surface_condition)
            &   (exited_box_condition)
            )

        detections_data = predictions_array[condition]
        detections_idx=torch.where(condition)[0]
       
        #[NumberOFDetection,86] detections_id , 85 of detections_data

        result = torch.cat((detections_idx.unsqueeze(1), detections_data), dim=1)
        result = sorted(result, key=lambda x: x [5] * max(x[6:]), reverse=True)

        # detected_features=None
        if  not return_only_detections :
            detected_features=[self.from_id_to_featureMap(id,features_array) for id in detections_idx]
        else:
            detected_features=[[] for _ in detections_idx]

        #  ([NumberOFDetection,86] , [128]) detections_id , 85 of detections
        return list(zip(result,detected_features))
        # ,detections_data[:,1:5]

    def convert_to_xyXY(self,box):
        x_center, y_center, w, h = box
        x = x_center - w / 2  # Calculate top-left x-coordinate
        y = y_center - h / 2  # Calculate top-left y-coordinate
        X = x_center + w / 2  # Calculate bottom-right x-coordinate
        Y = y_center + h / 2  # Calculate bottom-right y-coordinate
        return x, y, X, Y

    def iou(self,detection_1, detection_2):
        box1=self.convert_to_xyXY(detection_1 )
        box2=self.convert_to_xyXY(detection_2 )
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        iou = inter_area / (box1_area + box2_area - inter_area)
        return iou


    # def cluster_detections(self,detections, iou_threshold):
 
    #     clusters = []
    #     order_clusters = []
    #     # print(f"=>DATA cluster_detections : {detections}")
    #     for det in detections:
    #         assigned = False
    #         for cluster in clusters:
    #             if self.iou(det[0][1:5], cluster[0][0][1:5]) >= iou_threshold:
    #                 cluster.append(det)
    #                 assigned = True
    #                 break
    #         if not assigned:
    #             clusters.append([det])
     
    #     for obj in clusters:
    #         sorted_obj = sorted(obj, key=lambda x: x[0][5], reverse=True)
    #         # order_clusters.append(sorted_obj[:10])
    #         order_clusters.append(sorted_obj)

    #     return order_clusters
    ##################################
    def optimized_cluster_detections(self,detections, iou_threshold):
        clusters = []
        # order_clusters = []
        # print(f"=>Active Detections counts : {len(detections)}")
        for det in detections:
            assigned = False
            for cluster  in clusters:
                if self.iou(det[0][1:5], cluster[0][0][1:5]) >= iou_threshold:
                    assigned = True
                    if len(cluster)>self.max_anchors_number-1:
                        break
                    cluster.append(det)
                    break
            if not assigned :
                clusters.append([det])

        # print(f"self.max_anchor_count : {self.max_anchor_count}")
        # for i,obj in enumerate(clusters):
        #     print(f"obj {i} : {[x[0][5] for x in obj]}")
            
        return clusters
        
        # return order_clusters
 
    def bring_max_to_top(self,arr):
        max_index = np.argmax(arr[:,0,5])  # Find the index of the maximum element
        arr[0], arr[max_index] = arr[max_index], arr[0]  # Swap the maximum element with the first element
        return arr

    def detect_objects(self, frame,boxes_plotting=True):
        if self.network_input_size!=None and self.network_input_size != self.default_model_input_size:
            self.default_model_input_size=self.network_input_size
            print("UPDATE YOLO NETWORK INPUT SIZE ... ")
            # self.model.setInputParams(size=(self.default_model_input_size, self.default_model_input_size), scale=1/255, swapRB=True)
        start_time= time.perf_counter()
        # x_ratio=self.default_model_input_size/W
        # y_ratio=self.default_model_input_size/H

        with torch.no_grad():
            inference_start_time= time.perf_counter()
            # clusters,features=self.extract_objects_from_results(frame)

            start_time_= time.perf_counter()
            detections_results=self.extract_valide_objects(frame,class_num=[2,3,4,5,6,7,8],threshold=self.threshold,return_only_detections=boxes_plotting)
            # print(f"  yolo results times :  {time.perf_counter()-start_time_}")

            start_time_= time.perf_counter()
            detected_objects=self.optimized_cluster_detections(detections_results,iou_threshold=self.nms_threshold)
            # print(f"  optimized cluster detections times :  {time.perf_counter()-start_time_}")

            # start_time_= time.perf_counter()
            # self.cluster_detections(detections_results,iou_threshold=self.nms_threshold)
            # print(f"  clusters times :  {time.perf_counter()-start_time_}")
           
            start_time_= time.perf_counter()
            inference_time=np.round(time.perf_counter()-inference_start_time,4)
            if boxes_plotting:
                H , W=frame.shape[:2]
                x_ratio=self.default_model_input_size/W
                y_ratio=self.default_model_input_size/H
                for i  in range(len(detected_objects)): # 
                    # i=i+1
                # for j  in range(len(detected_objects[i])):
                    if len(detected_objects[i])==0:
                        continue
                    # get only the first box
                    anchor,feature = detected_objects[i][0]
                    x,y,w,h=anchor[1:5]
                    classConfidence = anchor[5]
                    classLabelID = np.argmax(anchor[6:])
                    classLabel = self.classesList[classLabelID]
                    classColor  = self.colors_list[classLabelID]

                    w=(int)(w/x_ratio)
                    h=(int)(h/y_ratio)
                    x=(int)(x/x_ratio-w/2)
                    y=(int)(y/y_ratio-h/2)

                    displayText = '{}: {:.2f} = {}'.format(classLabel, classConfidence,w*h)

                    cv2.rectangle(frame,(x,y),(x+w,y+h),color=classColor,thickness=2)
                    cv2.putText(frame, displayText, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)
                fps = 1 / np.round(time.perf_counter()-start_time,4)
                self.addFrameFps(frame,fps)
                print(f"  optimized detections times :  {time.perf_counter()-start_time_}")

                return frame,inference_time
            else: 
                return frame,detected_objects,self.classesList

 