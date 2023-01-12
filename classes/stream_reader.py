
from ast import Str
from curses import beep
from enum import Enum
from itertools import groupby
from math import floor
from operator import itemgetter
from select import select
import threading
import cv2,time,os, numpy as np
import pyshine as ps
from classes.detection_service import IDetectionService
from  classes.WebcamStream import WebcamStream
from  classes.RaspberryCamStream import RaspberryCamStream
from  classes.buffer import Buffer
from classes.background_subtractor_service import BackgroundSubtractorService
from utils_lib.enums import ClientStreamTypeEnum, StreamSourceEnum
import csv
import json
import base64
from csv import writer

class StreamReader: 

    nms_threshold=0.5
    threshold=0.5
    np.random.seed(123)
   
    def clean_memory(self):
        print("CALL DESTRUCTER FROM STREAM READER ")
        if  self.buffer:
            self.buffer.clean_memory()
            self.stop_reading_to_clean=True

    def __init__(self, detection_service:IDetectionService,stream_source:StreamSourceEnum, video_src:str ):
        self.perf = []
        self.classAllowed= []
        self.colorList= []
        self.inference_time_records = []
        self.fps_records = []
        
        self.current_selected_stream: ClientStreamTypeEnum=ClientStreamTypeEnum.CNN_DETECTOR
        self.current_batch=0
        self.stop_reading_from_user_action=True
        self.buffer :Buffer= None 
        self.current_time=0
        self.detection_service=detection_service
        self.detection_service.stream_reader=self
        self.current_frame_index=0
        self.stop_reading_for_loading=False
        self.end_of_file=False
        self.stream_source=stream_source
        self.video_src=video_src
        self.stop_reading_to_clean=False
        self.tracking_service=None
        self.background_subtractor_service=None
     

        if self.stream_source==StreamSourceEnum.FILE:
            self.buffer=Buffer(stream_source=StreamSourceEnum.FILE, video_src=video_src)
            print("StreamReader From File .....")
            self.buffer.stream_reader=self

        if self.stream_source==StreamSourceEnum.YOUTUBE:
            self.buffer=Buffer(stream_source=StreamSourceEnum.YOUTUBE, video_src=video_src)
            print("StreamReader From Youtube .....") 
            self.buffer.stream_reader=self

        if self.stream_source==StreamSourceEnum.WEBCAM:
            # self.webcam_stream = WebcamStream(src=self.video_src)
            # self.webcam_stream.start()
            self.raspberry_camera=RaspberryCamStream()
            self.raspberry_camera.picam2.start()
            print("StreamReader From Camera .....")
 

    def startBuffering(self):
        self.buffering_thread= threading.Thread(target=self.buffer.download_buffer)
        self.buffering_thread.start()
 
    def clean_streamer(self):
        print("================== >>STREAM_READER : begin clean_streamer")
        del(self.buffer.buffer_frames)
        self.buffer.cap.release()
        del(self.buffer.cap)
        print("STREAM_READER : end clean_streamer")

    def reset(self):
        if self.stream_source == StreamSourceEnum.YOUTUBE :
            self.buffer.reset_youtube_buffer(youtube_url=self.video_src)   
        if self.stream_source == StreamSourceEnum.FILE:
            self.buffer.reset_file_buffer(file_src=self.video_src)   
        self.current_time=0
        self.current_sec=0
        self.current_frame_index=0

    def read_stream(self):
        if self.stream_source == StreamSourceEnum.WEBCAM:   
            # yield from self.read_stream_from_webcam()
            yield from self.read_stream_from_raspberry_cam()
        else:
            yield from self.read_stream_from_buffer()
        return "NO STREAM TO READ"


    def read_stream_from_raspberry_cam(self):
        print("Start READING FROM raspberry webcam ......")
        

        while(True):
            image = self.raspberry_camera.picam2.capture_array()
            if self.stop_reading_from_user_action:
                time.sleep(.2)
                break
            yield from self.ProcessAndYieldFrame(image,3)
        # for frame in self.raspberry_camera.camera.capture_continuous(self.raspberry_camera.rawCapture, format="bgr", use_video_port=True):
        #     if self.stop_reading_from_user_action:
        #         time.sleep(.2)
        #         break
        #     image = frame.array
        #     yield from self.ProcessAndYieldFrame(image,3)
        #     self.raspberry_camera.rawCapture.truncate(0)
 

    def read_stream_from_webcam(self):
        # delay for buffer to load some frame

        detection_fps=1
        print("Start READING FROM webcam ......")
        refresh_rate=1
        counter=0
        start_time = time.perf_counter() 
        while True: 
            # IF BUTTON STOP PRESSED CONTINUE
            if self.stop_reading_from_user_action:
                time.sleep(.2)
                continue
           
            # ret,frame=self.webcam_stream.read()    

            # if self.detection_service !=None and self.detection_service.get_selected_model() !=None:
            #     frame , inference_time= self.detection_service.detect_objects(frame, threshold= self.threshold ,nms_threshold=self.nms_threshold)
            
            # frame =  ps.putBText(frame,str(detection_fps)+" fps",text_offset_x=320,text_offset_y=20,vspace=10,hspace=10, font_scale=1.2,text_RGB=(255,25,50))
            # ret,buffer=cv2.imencode('.jpg',frame)
            # frame=buffer.tobytes()
            # yield from self.yieldSelectedStream(detection_fps)
            frame=self.getNextFrame()
            yield from self.ProcessAndYieldFrame(frame,detection_fps)
            
            counter+=1
            if (time.perf_counter() - start_time) > refresh_rate :
                detection_fps= round(counter / (time.perf_counter() - start_time),2)
                counter = 0
                start_time = time.perf_counter()
            
    def read_stream_from_buffer(self):
        # delay for buffer to load some frame
        time.sleep(0.1)
        self.current_sec=0
        t1= time.perf_counter()
        detection_fps=0
        current_fps=0
        jump_frame=0    
        print("Start READING FROM BUFFER ......")
        diff_time=0
        # backgroundSubtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

        while True :
            t1= time.perf_counter()
            
            if self.stop_reading_to_clean:
                break
            # IF BUTTON STOP PRESSED CONTINUE
            if self.stop_reading_from_user_action:
                time.sleep(.2)
                continue
            
             # END READING IN CASE OF REACHING THE LAST BATCH
            if ( self.buffer.last_frame_of_last_batch==self.current_frame_index  and self.current_batch==self.buffer.last_batch):
                continue;   

            if self.current_frame_index>=len(self.buffer.buffer_frames) or len(self.buffer.buffer_frames)==0 :
                continue
            
            # RETURN FRAMES TO NAV
            frame=self.getNextFrame()
            yield from self.ProcessAndYieldFrame(frame,detection_fps)
            
            # ADD SECONDES
            self.current_time=self.current_time+self.buffer.frame_duration*(floor(jump_frame))
            new_sec=self.current_time
            if new_sec>=self.current_sec+1:
                self.current_sec=self.current_sec+1
                new_sec=self.current_sec
                detection_fps=current_fps
            # GO to THE NEXT FRAME
            self.current_frame_index=self.current_frame_index + floor(jump_frame) 
            
            # sleep if the FPS is too high 
            if current_fps > self.buffer.fps:
                time.sleep(0.01)
         
            jump_frame=jump_frame-floor(jump_frame)

            # DELETE PREVIOUS BATCH FRAMES ALREADY PRINTED FROM MEMORY
            if (self.current_frame_index>=self.buffer.batch_size):
                self.buffer.delete_last_batch(self.current_batch)
                self.current_frame_index=0
                self.buffer.download_new_batch=True
        
            diff_time=time.perf_counter()-t1    
            # jump frames in function of processing time consumtion to simulate real time detection
            jump_frame=jump_frame+diff_time/self.buffer.frame_duration
          
            # UPDATE FPS
            current_fps=round(1/diff_time,2)

            # SAVE RECORDES CSV
            # if self.detection_service !=None  and self.detection_service.get_selected_model() !=None:
            # self.records.append({'detector': self.detection_service.get_selected_model()['name'],'fps':float(1/diff_time) ,'inference_time':inference_time})
            self.fps_records.append(float(1/diff_time) )
          
        
        print(" :::: END STREAM READER LOOP")

    def getCurrentFrame(self):   

        if self.stream_source ==StreamSourceEnum.FILE:
            origin_frame,self.current_batch= self.buffer.buffer_frames[self.current_frame_index] 
        elif self.stream_source == StreamSourceEnum.WEBCAM:
            ret,origin_frame=self.webcam_stream.read()  

        return origin_frame,self.current_batch

    def applyDetection(self,origin_frame):   
        if self.detection_service !=None  and self.detection_service.get_selected_model() !=None:
            detection_frame ,inference_time = self.detection_service.detect_objects(origin_frame, threshold= self.threshold ,nms_threshold=self.nms_threshold)
            return detection_frame,inference_time
        return origin_frame,-1

    # def applyBackgroundSubtraction(self,origin_frame):
    #     return self.background_subtractor_service.apply(origin_frame)

    def encodeStreamingFrame(self,frame,resize_ratio=1,jpeg_quality=100):
            if resize_ratio!=1:
                frame=cv2.resize(frame, (int(self.buffer.width*resize_ratio) ,int(self.buffer.height*resize_ratio) ))
            ret,buffer=cv2.imencode('.jpg',frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            img_bytes=buffer.tobytes()
            return  base64.b64encode(img_bytes).decode()

    def save_records(self):
        if self.detection_service !=None  and self.detection_service.get_selected_model() !=None:
            csv_file = "results.csv"
            average_fps=0    
            average_inference_time=0    
            with open(csv_file, 'a') as csvfile:
                writer_object = writer(csvfile)                
                for index, item in enumerate(self.fps_records):
                    if index==0:
                        continue
                    average_fps+=item

                for index, item in enumerate(self.inference_time_records):
                    if index==0:
                        continue
                    average_inference_time+=item

                average_fps/=(len(self.fps_records)-1)
                average_inference_time/=(len(self.inference_time_records)-1)
                writer_object.writerow([self.detection_service.get_selected_model()['name'],average_fps,average_inference_time])
                self.inference_time_records=[]
                self.fps_records=[]

                 
    def addFrameFpsAndTime(self,frame,detection_fps):
        if self.stream_source!=StreamSourceEnum.WEBCAM :
            frame =  ps.putBText(frame,str( "{:02d}".format(self.current_sec//60))+":"+str("{:02d}".format(self.current_sec%60)),text_offset_x=20,text_offset_y=20,vspace=10,hspace=10, font_scale=1.4,text_RGB=(255,255,250))
        # frame =  ps.putBText(frame,str(detection_fps)+" fps",text_offset_x=320,text_offset_y=20,vspace=10,hspace=10, font_scale=1.2,text_RGB=(255,25,50))
        # return frame

    def addInferenceTime(self, frame, inferenceTime):
        color = 2**16-1
        frame = cv2.putText(frame,str( "inf. time : {:02f}".format(inferenceTime)), (60, 45), cv2.FONT_HERSHEY_SIMPLEX, 3.3, color, 1, cv2.LINE_AA)


    def change_video_file(self,new_video_file): 
        if self.stream_source == StreamSourceEnum.FILE:
            self.video_src=new_video_file
            self.buffer.reset_file_buffer(file_src=self.video_src)   
            self.current_time=0
            self.current_sec=0
            self.current_frame_index=0
    
    def getNextFrame(self):
        origin_frame,self.current_batch=self.getCurrentFrame() 
        return origin_frame

    def ProcessAndYieldFrame(self,frame,detection_fps):
        result={}
        copy_frame=frame.copy()

        if self.current_selected_stream== ClientStreamTypeEnum.CNN_DETECTOR:
            # origin_frame,self.current_batch=self.getCurrentFrame() 
            detection_frame,inference_time=self.applyDetection(copy_frame)
            self.inference_time_records.append(inference_time)
            self.addFrameFpsAndTime(detection_frame,detection_fps)
            result['detectorStream']=self.encodeStreamingFrame(frame=detection_frame,resize_ratio=1,jpeg_quality=80)

        elif self.current_selected_stream== ClientStreamTypeEnum.BACKGROUND_SUBTRACTION:
            foreground_detection_frame,raw_mask_frame,inference_time=self.background_subtractor_service.apply(copy_frame)
            # self.addInferenceTime(subtraction_frame,inference_time)
            self.addFrameFpsAndTime(foreground_detection_frame,detection_fps)
            result['backgroundSubStream_1']=self.encodeStreamingFrame(frame=raw_mask_frame,resize_ratio=1,jpeg_quality=80)
            result['backgroundSubStream_2']=self.encodeStreamingFrame(frame=foreground_detection_frame,resize_ratio=1,jpeg_quality=80)
        
        elif self.current_selected_stream== ClientStreamTypeEnum.TRACKING_STREAM:
            tracking_frame,inference_time=self.tracking_service.apply(copy_frame)
            result['trackingStream_1']=self.encodeStreamingFrame(frame=tracking_frame,resize_ratio=1,jpeg_quality=80)
            # result['trackingStream_2']=self.encodeStreamingFrame(frame=tracking_frame,resize_ratio=1,jpeg_quality=80)
            # result['testStream_first']=self.test_stream(origin_frame.copy(),detection_fps)
           
        yield 'event: message\ndata: ' + json.dumps(result) + '\n\n'

    # i=0
    # def test_stream(self,frame,detection_fps):

    #     # frame,inference_time=self.applyDetection(frame)
       
    #     height, width = frame.shape[:2]
    #     # (x, y, w, h) = (int) (100), (int)(height/4),  (int)(width)-200 ,150
    #     (x, y, w, h) =    400 , 200,  160 , 160
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), 2**16-1, 2)
    
    #     detection_region = frame[y:y+h, x:x+w]

    #     self.i+=1
    #     if self.i%20 ==0 :
    #         detection_region,inference_time=self.applyDetection(detection_region)

    #     white_rect = np.zeros(detection_region.shape, dtype=np.uint8) 
    #     white_rect[:, :, 0] = 255

    #     res = cv2.addWeighted(detection_region, 1, white_rect, 0.5, 1.0)
    #     frame[y:y+h, x:x+w] = res
    #     cv2.rectangle(frame, (x, y), (x + w, y + h),  (255, 0, 0), 1)

    #     self.addFrameFpsAndTime(frame,detection_fps)
    #     return self.encodeStreamingFrame(frame=frame,resize_ratio=1,jpeg_quality=90)
     
    
    
    
    # less /proc/cpuinfo
