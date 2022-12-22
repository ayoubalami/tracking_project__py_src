
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
from  classes.buffer import Buffer
from utils_lib.enums import StreamSourceEnum
import csv




class StreamReader: 

    np.random.seed(123)
    # threshold = 0.5   
    # nms_threshold =0.5
    perf = []
    classAllowed=[]
    colorList=[]
    buffer :Buffer= None
    records =[]

    def clean_memory(self):
        print("CALL DESTRUCTER FROM STREAM READER ")
        if  self.buffer:
            self.buffer.clean_memory()
            self.stop_reading_to_clean=True

    def __init__(self, detection_service:IDetectionService,stream_source:StreamSourceEnum, video_src:str,threshold:float,nms_threshold:float):
        self.stop_reading_from_user_action=True
        self.current_time=0
        self.detection_service=detection_service
        self.detection_service.stream_reader=self
        self.current_frame_index=0
        self.stop_reading_for_loading=False
        self.end_of_file=False
        self.stream_source=stream_source
        self.video_src=video_src
        self.stop_reading_to_clean=False
        self.nms_threshold=nms_threshold
        self.threshold=threshold

        if self.stream_source==StreamSourceEnum.FILE:
            self.buffer=Buffer(stream_source=StreamSourceEnum.FILE, video_src=video_src)
            print("StreamReader From File .....")

        if self.stream_source==StreamSourceEnum.YOUTUBE:
            self.buffer=Buffer(stream_source=StreamSourceEnum.YOUTUBE, video_src=video_src)
            print("StreamReader From Youtube .....")
     
        if self.stream_source==StreamSourceEnum.WEBCAM:
            self.webcam_stream = WebcamStream(src=self.video_src)
            self.webcam_stream.start()
            print("StreamReader From Camera .....")

        self.buffer.stream_reader=self
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
            yield from self.read_stream_from_webcam()
        else:
            yield from self.read_stream_from_buffer()

        return "NO STREAM TO READ"


    def read_stream_from_webcam(self):
          # delay for buffer to load some frame

        detection_fps=1
        print("Start READING FROM webcam ......")
        refresh_rate=1
        counter=0
        start_time = time.time() 
        while True: 

            # IF BUTTON STOP PRESSED CONTINUE
            if self.stop_reading_from_user_action:
                time.sleep(.2)
                continue
            ret,frame=self.webcam_stream.read()    

            if self.detection_service !=None:
                if self.detection_service.get_selected_model() !=None:
                    frame = self.detection_service.detect_objects(frame, threshold= self.threshold ,nms_threshold=self.nms_threshold)
            
            # frame =  ps.putBText(frame,str( "{:02d}".format(self.current_sec//60))+":"+str("{:02d}".format(self.current_sec%60)),text_offset_x=20,text_offset_y=20,vspace=10,hspace=10, font_scale=1.4,text_RGB=(255,255,250))
            frame =  ps.putBText(frame,str(detection_fps)+" fps",text_offset_x=320,text_offset_y=20,vspace=10,hspace=10, font_scale=1.2,text_RGB=(255,25,50))

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
         
            counter+=1
            if (time.time() - start_time) > refresh_rate :
                detection_fps= round(counter / (time.time() - start_time))
                counter = 0
                start_time = time.time()
            
    def read_stream_from_buffer(self):
        # delay for buffer to load some frame
        time.sleep(0.1)
        self.current_sec=0
        t1= time.time()
        detection_fps=0
        current_fps=0
        jump_frame=0    
        print("Start READING FROM BUFFER ......")
        diff_time=0
        while True :
            t1= time.time()
            
            if self.stop_reading_to_clean:
                break
            # IF BUTTON STOP PRESSED CONTINUE
            if self.stop_reading_from_user_action:
                time.sleep(.2)
                continue
            
             # END READING IN CASE OF REACHING THE LAST BATCH
            if ( self.buffer.last_frame_of_last_batch==self.current_frame_index  and current_batch==self.buffer.last_batch):
                # self.clean_streamer()
                # print("STREAM_READER : end_of_file")
                continue;   


            if self.current_frame_index>=len(self.buffer.buffer_frames) or len(self.buffer.buffer_frames)==0 :
                continue
            
            
            # SENT FRAMES TO NAV
            frame,current_batch,inference_time=self.getCurrentFrame() 
            
            frame =  ps.putBText(frame,str( "{:02d}".format(self.current_sec//60))+":"+str("{:02d}".format(self.current_sec%60)),text_offset_x=20,text_offset_y=20,vspace=10,hspace=10, font_scale=1.4,text_RGB=(255,255,250))
            frame =  ps.putBText(frame,str(detection_fps)+" fps",text_offset_x=320,text_offset_y=20,vspace=10,hspace=10, font_scale=1.2,text_RGB=(255,25,50))
            # print("show frame...  !! ", self.current_sec ," :: ", self.current_time)
            ret,buffer=cv2.imencode('.jpg',frame)

            img_bytes=buffer.tobytes()

            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')
            
            # JUMP N FRAMES 
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
                self.buffer.delete_last_batch(current_batch)
                self.current_frame_index=0
                self.buffer.download_new_batch=True
        
 
            diff_time=time.time()-t1    
            # jump frames in function of processing time consumtion to simulate real time detection
            jump_frame=jump_frame+diff_time/self.buffer.frame_duration
          
            self.records.append({'detector': self.detection_service.get_selected_model()['name'],'fps':float(1/diff_time) ,'inference_time':inference_time})
            current_fps=round(1/diff_time)
        
        print(" :::: END STREAM READER LOOP")

            # print(" ",diff_time ," ; ",jump_frame," = ",fps)

    def getCurrentFrame(self):   
        frame,current_batch= self.buffer.buffer_frames[self.current_frame_index] 
        if self.detection_service !=None:
            if self.detection_service.get_selected_model() !=None:
                frame ,inference_time = self.detection_service.detect_objects(frame, threshold= self.threshold ,nms_threshold=self.nms_threshold)
        return frame,current_batch,inference_time
        
    def save_records(self):
        csv_columns = ['detector','fps','inference_time']
        csv_file = "/root/shared/records/"+self.detection_service.get_selected_model()['name']+".csv"
        average_fps=0    
        average_inference_time=0    
           
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for index, item in enumerate(self.records):
                if index==0:
                    continue
                writer.writerow(item)
                average_fps+=item['fps']
                average_inference_time+=item['inference_time']

            writer.writerow({'detector':self.detection_service.get_selected_model()['name']+"-average",'fps':average_fps/(len(self.records)-1),'inference_time':average_inference_time/(len(self.records)-1)})
            self.records= [] 

    # less /proc/cpuinfo
