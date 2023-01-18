import cv2
from classes.detection_service import IDetectionService
from utils_lib.enums import StreamSourceEnum
import math

class OfflineTracker:
    nms_threshold=0.5
    threshold=0.5

    def __init__(self, detection_service:IDetectionService,stream_source:StreamSourceEnum, video_src:str):
        self.detection_service=detection_service
        self.video_src=video_src 
        self.stream_source=stream_source
        self.cap=None
        # out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
        if self.stream_source==StreamSourceEnum.FILE:
            print("Record From File service loaded .....")
            self.cap= cv2.VideoCapture(video_src)
        width= int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height= int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_length = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # self.video_writer= cv2.VideoWriter('/root/shared/out.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (int(width ),int(height )))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_filename='out.mp4'
        if self.detection_service.get_selected_model() !=None:
            output_filename=self.detection_service.get_selected_model()['name']+"-out.mp4"
        self.video_writer = cv2.VideoWriter('/root/shared/out_videos/'+output_filename, fourcc, 20, (width, height), True)
 
    def start(self):
        print("Record started .....")
        i=0
        last_frame=300
        # last_frame=self.video_length
        step=math.ceil(last_frame/10)
        while True:
            success, frame = self.cap.read()
            if self.detection_service !=None and self.detection_service.get_selected_model()!=None:
                frame , _= self.detection_service.detect_objects(frame, threshold= self.threshold ,nms_threshold=self.nms_threshold)
                
            if success == False or i > last_frame :
                break
            i=i+1
            if i%step==0:
                print(">>>"+str((int)((i*100)/last_frame))+"% video progress ...")
                
            self.video_writer.write(frame)

        self.cap.release()
        self.video_writer.release()
        cv2.destroyAllWindows()
        print("Record done .")


