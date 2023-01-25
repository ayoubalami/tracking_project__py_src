
from asyncio import StreamReader
import cv2


class IDetectionService:
    stream_reader:StreamReader
    selected_model=None
    
    def init_selected_model(self):
        self.selected_model=None

    def get_selected_model(self):
        pass

    def service_name(self):
        pass

    def load_model(self,model=None):
        pass

    def detect_objects(self, frame,threshold:float,nms_threshold:float,boxes_plotting=True,network_input_size=None):
        pass

    def get_object_detection_models(self):
        pass

    def clean_memory(self):
        pass

    def addFrameFps(self,img,detection_fps):
        width=img.shape[1]
        cv2.putText(img, f'FPS: {round(detection_fps,2)}', (int(width/2)-20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,25,50), 2)
    