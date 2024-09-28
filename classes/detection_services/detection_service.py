
import cv2
import random

class IDetectionService:
    stream_reader=None
    selected_model=None
    network_input_size=416
    # min_surface_area=2500
    # network_input_size=416
    allowed_classes=[]
    colors_list=[]
    threshold=0.4
    nms_threshold=0.3
    CNN_video_resolution_ratio=1
    # max_anchor_count=10
    
    def init_selected_model(self):
        self.selected_model=None

    def get_selected_model(self):
        return self.selected_model

    def service_name(self):
        pass

    def load_model(self,model=None):
        pass

    def detect_objects(self, frame,boxes_plotting=True):
        pass

    def clean_memory(self):
        pass

    def init_object_detection_models_list(self):
        self.detection_method_list_with_url=self.detection_method_list

    def get_object_detection_models(self):
        return self.detection_method_list 

    def addFrameFps(self,img,detection_fps):
        # pass
        width=img.shape[1] 
        cv2.putText(img, f'FPS: {round(detection_fps,2)}', (int(width/2)-20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,25,50), 2)
        # cv2.putText(img, f'FPS: {round(detection_fps,2)}', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,25,50), 2)
    
    def readClasses(self): 
        with open(self.classFile, 'r') as f:
            self.classesList = f.read().splitlines()
        print(self.classesList)
        # set static Colors of 8 first classes
        self.colors_list =    [[23, 213, 104],
                            [168, 240,  62],
                            [  3,   6, 240],
                            [235, 119,  95],
                            [138, 219, 166],
                            [ 59, 197,  34],
                            [ 42, 156,  60],
                            [ 142, 56,  250]]
        # add random colors to remaining colors
        more_random_color=[[random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)] for i in range(73)] 
        self.colors_list+=more_random_color
        
    def keepSelectedClassesOnly(self,bboxs,confidences,class_idx,threshold):
        condidats=list(zip(bboxs,confidences,class_idx))
        if len(self.allowed_classes)==0 :
            allowed_condidats = [condidat for  condidat in condidats if condidat[1]>=threshold]
            return allowed_condidats
        allowed_condidats = [condidat for  condidat in condidats if condidat[2] in self.allowed_classes and condidat[1]>=threshold]
        if len(allowed_condidats)>0:
            return allowed_condidats
        return None

    def resize_frame(self,frame):
        img_height,img_width = frame.shape[:2] 
        if self.CNN_video_resolution_ratio<1:
            return cv2.resize(frame, (int(img_width*self.CNN_video_resolution_ratio) ,int(img_height*self.CNN_video_resolution_ratio) ))
        return frame