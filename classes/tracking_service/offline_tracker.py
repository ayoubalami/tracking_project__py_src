import cv2
from classes.tracking_service.tracking_service import TrackingService
from utils_lib.enums import StreamSourceEnum
import math

class OfflineTracker:
    nms_threshold=0.5
    threshold=0.5

    def __init__(self, tracking_service:TrackingService,stream_source:StreamSourceEnum, video_src:str):
        self.tracking_service=tracking_service
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
        video_name=(video_src.split("/")[-1]).split(".")[0]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_filename=video_name+'no_detector-track_out.mp4'
        if self.tracking_service.detection_service and self.tracking_service.detection_service.get_selected_model() !=None:
            output_filename=video_name+"-"+self.tracking_service.detection_service.get_selected_model()['name']+'_'+str(self.threshold)+"-track_out.mp4"
        self.video_writer = cv2.VideoWriter('/root/shared/out_videos/'+output_filename, fourcc, 20, (width, height), True)

    def start(self):
        print("Record started tracking_offline .....")
        print(str(self.threshold)+" threshold")
        print(str(self.nms_threshold)+" nms threshold")
        i=0
        max_out_frames=2400
        step=math.ceil(max_out_frames/10)
        while True:
            success, frame = self.cap.read()
            if success == False or i > max_out_frames :
                break
            if self.tracking_service !=None :
                # if self.tracking_service.detection_service and self.tracking_service.detection_service.get_selected_model()!=None:
                frame=self.tracking_service.apply(frame,threshold= self.threshold ,nms_threshold=self.nms_threshold)   
            
            i=i+1
            if i%step==0:
                print(">>>"+str((int)((i*100)/max_out_frames))+"% video progress ...")
            self.video_writer.write(frame)
        self.cap.release()
        self.video_writer.release()
        cv2.destroyAllWindows()
        print("Record done .")