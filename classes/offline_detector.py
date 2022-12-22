import cv2
from classes.detection_service import IDetectionService
from utils_lib.enums import StreamSourceEnum
 
class OfflineDetector:
    def __init__(self, detection_service:IDetectionService,stream_source:StreamSourceEnum, video_src:str,threshold:float,nms_threshold:float ):

        self.detection_service=detection_service
        self.video_src=video_src
        self.nms_threshold=nms_threshold
        self.threshold=threshold
        self.stream_source=stream_source
        self.cap=None
        # out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
        if self.stream_source==StreamSourceEnum.FILE:
            print("Record From File service loaded .....")
            self.cap= cv2.VideoCapture(video_src)
        width= int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height= int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # self.video_writer= cv2.VideoWriter('/root/shared/out.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (int(width ),int(height )))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter('/root/shared/out.mp4', fourcc, 20, (width, height), True)
 
    def start(self):
        print("Record started .....")
        i=0
        while True:
            success, frame = self.cap.read()
            if success == False or i > 1000 :
                break
            i=i+1
            self.video_writer.write(frame)

        self.cap.release()
        self.video_writer.release()
        cv2.destroyAllWindows()
        print("Record done .")


