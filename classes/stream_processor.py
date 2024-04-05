

# from app_service  import AppService
import cv2,time,os, numpy as np
from classes.detection_services.detection_service import IDetectionService
from  classes.webcam_reader import WebcamReader
# from  classes.buffer import Buffer
from classes.background_subtractor_service import BackgroundSubtractorService
from classes.tracking_service.tracking_service import TrackingService
from classes.hybrid_tracking_service.hybrid_tracking_service import HybridTrackingService
from classes.video_stream import VideoStream
from utils_lib.enums import  StreamSourceEnum
import json
import base64
from utils_lib.enums import  ProcessingTaskEnum

class StreamProcessor: 

    np.random.seed(123)
    detection_service :IDetectionService= None
    background_subtractor_service: BackgroundSubtractorService=None
    tracking_service: TrackingService=None
    hybrid_tracking_service: HybridTrackingService=None
    video_stream:VideoStream
    stream_source : StreamSourceEnum
    processing_task:ProcessingTaskEnum=ProcessingTaskEnum.RAW_STREAM
    video_resolution_ratio=1
    jpeg_quality=80
    current_fps=0
    saveOutputVideo=True
    startRecording=False

    def __init__(self,stream_source:StreamSourceEnum,detection_service,background_subtractor_service,tracking_service,hybrid_tracking_service):
       
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_filename='out_tracker'+str(time.perf_counter())+'.mp4'
        self.video_writer = cv2.VideoWriter('/root/shared/out_video_tracker/'+output_filename, fourcc, 20, (1192, 848), True)
 
        self.stream_source=stream_source        
        if self.stream_source==StreamSourceEnum.FILE:
            self.video_stream=VideoStream(self.stream_source)
        elif self.stream_source==StreamSourceEnum.WEBCAM:
            self.webcam_reader = WebcamReader(src="http://10.10.23.14:9000/video")
     
        self.detection_service=detection_service
        self.background_subtractor_service=background_subtractor_service
        self.tracking_service=tracking_service
        self.hybrid_tracking_service=hybrid_tracking_service
        
    def return_stream(self):
        if self.stream_source==StreamSourceEnum.FILE:
            yield from self.return_video_stream()
        elif self.stream_source==StreamSourceEnum.WEBCAM:
            yield from self.return_webcam_stream()

    def return_video_stream(self):
        
        self.start_time=time.perf_counter() 
        for frame in self.video_stream.get_frames():
            if not self.video_stream.stopped or self.video_stream.one_next_frame:
                result=self.process_frame(frame)

                if self.saveOutputVideo:
                    if self.video_stream.stopped and  self.startRecording :
                        self.video_writer.release()
                    else:
                        self.video_writer.write(frame)
                        self.startRecording=True

            yield 'event: message\ndata: ' + json.dumps(result) + '\n\n'
            
    def return_webcam_stream(self):
        self.start_time=time.perf_counter() 
        while True:
            ret,frame=self.webcam_reader.read()  
            result=self.process_frame(frame)
            yield 'event: message\ndata: ' + json.dumps(result) + '\n\n' 

    def process_frame(self,frame):
        result={}  
        if self.processing_task== ProcessingTaskEnum.RAW_STREAM:
            result['detectorStream']=self.encodeStreamingFrame(frame=frame)
        elif self.processing_task== ProcessingTaskEnum.CNN_DETECTOR:
            result= self.process_detection(frame)
        elif self.processing_task== ProcessingTaskEnum.BACKGROUND_SUBTRACTION:
            result= self.process_bs(frame)
        elif self.processing_task== ProcessingTaskEnum.TRACKING_STREAM:
            result= self.process_tracking(frame)
        elif self.processing_task== ProcessingTaskEnum.HYBRID_TRACKING_STREAM:
            result= self.process_hybrid_tracking(frame)
        self.current_fps=1/(time.perf_counter()-self.start_time+(1e-6)) 
        self.start_time=time.perf_counter() 
        return result

    def process_detection(self,frame):
        result={}
        if self.detection_service !=None and self.detection_service.get_selected_model() !=None:
            detection_frame ,inference_time = self.detection_service.detect_objects(frame)
            result['detectorStream']=self.encodeStreamingFrame(frame=detection_frame)
            return result
        result['detectorStream']=self.encodeStreamingFrame(frame=frame)
        return result

    def process_bs(self,frame):
        result={}
        merged_foreground_detection_frame,resized_foreground_detection_frame,raw_mask_frame,inference_time=self.background_subtractor_service.apply(frame=frame)
        result['backgroundSubStream_1']=self.encodeStreamingFrame(frame=raw_mask_frame)
        result['backgroundSubStream_2']=self.encodeStreamingFrame(frame=resized_foreground_detection_frame)
        result['backgroundSubStream_3']=self.encodeStreamingFrame(frame=merged_foreground_detection_frame)
        return result

    def process_tracking(self,frame):
        result={}
        tracking_frame=self.tracking_service.apply(frame)
        result['trackingStream_1']=self.encodeStreamingFrame(frame=tracking_frame)
        return result

    def process_hybrid_tracking(self,frame):
        result={}
        hybrid_tracking_frame=self.hybrid_tracking_service.apply(frame)
        result['hybridTrackingStream_1']=self.encodeStreamingFrame(frame=hybrid_tracking_frame)
        return result

    def resize_frame(self,frame):
        if self.video_resolution_ratio!=1:
            resized_frame=cv2.resize(frame.copy(), (int(self.video_stream.width*self.video_resolution_ratio) ,int(self.video_stream.height*self.video_resolution_ratio) ))
            return frame,resized_frame
        return frame,frame

    def encodeStreamingFrame(self,frame):
        resized_frame=frame
        self.add_frame_time(frame)
        self.add_fps(frame)
        if self.jpeg_quality!=100:
            ret,buffer=cv2.imencode('.jpg',resized_frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        else:
            ret,buffer=cv2.imencode('.jpg',resized_frame)
        img_bytes=buffer.tobytes()
        return  base64.b64encode(img_bytes).decode()

    def add_frame_time(self,frame):
        if self.stream_source==StreamSourceEnum.FILE :
            minute,second=((int)(self.video_stream.current_video_time))//60,((int)(self.video_stream.current_video_time))%60
            time=str("{:02d}".format(minute))+":"+str( "{:02d}".format(second))
            cv2.putText(frame,time, ((int)(20*self.video_resolution_ratio), (int)(60*self.video_resolution_ratio)), cv2.FONT_HERSHEY_SIMPLEX, (float)(1.4*self.video_resolution_ratio), (255,255,250),  (2), cv2.LINE_AA)

    def add_fps(self,frame): 
        pass
        # if self.current_fps !=0 and self.stream_source==StreamSourceEnum.FILE  and self.processing_task!= ProcessingTaskEnum.RAW_STREAM:
        #     fps=str("FPS : {:02.1f}".format(self.current_fps))
        #     cv2.putText(frame,fps, ((int)(self.video_stream.width*self.video_resolution_ratio/2 -(100*self.video_resolution_ratio)), (int)(50*self.video_resolution_ratio)), cv2.FONT_HERSHEY_SIMPLEX, (float)(1.2*self.video_resolution_ratio), (255,55,50),  (2), cv2.LINE_AA)

    def reset(self):
        self.video_resolution_ratio=1
        # self.processing_task=ProcessingTaskEnum.RAW_STREAM
        self.video_stream.reset()
        self.hybrid_tracking_service.reset()

    def start(self,selected_video):
        if self.stream_source==StreamSourceEnum.FILE:
            if (self.video_stream.video_file!=selected_video):
                self.video_stream.video_file=selected_video
            self.video_stream.start()
        elif self.stream_source==StreamSourceEnum.WEBCAM:
            self.webcam_reader = WebcamReader(src="http://10.10.23.14:9000/video")
    
    def stop(self):
        if self.stream_source==StreamSourceEnum.FILE:
            self.video_stream.stop()
        elif self.stream_source==StreamSourceEnum.WEBCAM:
            self.webcam_reader.stop()