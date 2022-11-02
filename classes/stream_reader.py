
from ast import Str
from math import floor
import threading
import cv2,time,os, numpy as np
from classes.buffer import Buffer
import pyshine as ps
from classes.detection_service import IDetectionService
import tensorflow as tf

class StreamReader: 

    np.random.seed(123)
    # threshold = 0.5
    perf = []
    classAllowed=[]
    colorList=[]

    def __init__(self, buffer:Buffer,detection_service:IDetectionService):
       
        # self.target_delay=buffer.frame_duration 
        self.buffer=buffer
        self.detection_service=detection_service
        self.current_time=0
        self.current_frame_index=0
        self.stop_reading_for_loading=False
        self.stop_reading_from_user_action=True
        self.end_of_file=False
        self.buffer.stream_reader=self
        # incrementFrameThread= threading.Thread(target=self.incrementFrame)
        # incrementFrameThread.start()
        # self.last_ticke_time=0
        # self.detectionModule.load_model()

    def clean_streamer(self):
        print("STREAM_READER : begin clean_streamer")
        del(self.buffer.buffer_frames)
        self.buffer.cap.release()
        del(self.buffer.cap)
        print("STREAM_READER : end clean_streamer")

    def readStream(self):
        # delay for buffer to load some frame
        time.sleep(0.1)
        current_sec=0
        t1= time.time()
        detection_fps=0
        current_fps=0
        jump_frame=0    
        print("Start READING ......")
        diff_time=0
        while True :
            
            # print("Start LOOPING READING ......")

            t1= time.time()
            # IF BUTTON STOP PRESSED CONTINUE
            if self.stop_reading_from_user_action:
                time.sleep(.2)
                continue

            if (self.buffer.stop_buffring_event.is_set()):
                self.clean_streamer()
                print("STREAM_READER : end_of_thread")
                break; 

            if self.current_frame_index>=len(self.buffer.buffer_frames) or len(self.buffer.buffer_frames)==0 :
                continue
            
            # SENT FRAMES TO NAV
            frame,current_batch=self.getCurrentFrame() 
            
            frame =  ps.putBText(frame,str( "{:02d}".format(current_sec//60))+":"+str("{:02d}".format(current_sec%60)),text_offset_x=20,text_offset_y=20,vspace=10,hspace=10, font_scale=1.4,text_RGB=(255,255,250))
            frame =  ps.putBText(frame,str(detection_fps)+" fps",text_offset_x=320,text_offset_y=20,vspace=10,hspace=10, font_scale=1.2,text_RGB=(255,25,50))
            
            ret,buffer=cv2.imencode('.jpg',frame)

            img_bytes=buffer.tobytes()

            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')
            

            # JUMP N FRAMES 
            # ADD SECONDES
            self.current_time=self.current_time+self.buffer.frame_duration*(floor(jump_frame))
            new_sec=self.current_time
            if new_sec>=current_sec+1:
                current_sec=current_sec+1
                new_sec=current_sec
                detection_fps=current_fps
            # GO to THE NEXT FRAME
            self.current_frame_index=self.current_frame_index + floor(jump_frame) 
            # if(diff_time>0):
            #     detection_fps= jump_frame/diff_time 

            jump_frame=jump_frame-floor(jump_frame)

            # DELETE PREVIOUS BATCH FRAMES ALREADY PRINTED FROM MEMORY
            if (self.current_frame_index>=self.buffer.batch_size):
                self.buffer.delete_last_batch(current_batch)
                self.current_frame_index=0
                self.buffer.download_new_batch=True
            
            # END READING IN CASE OF REACHING THE LAST BATCH
            if ( self.buffer.last_frame_of_last_batch==self.current_frame_index  and current_batch==self.buffer.last_batch):
                self.clean_streamer()
                print("STREAM_READER : end_of_file")
                break;   
 
            # time.sleep(0.005)

            diff_time=time.time()-t1    
            jump_frame=jump_frame+diff_time/self.buffer.frame_duration
            current_fps=round(1/diff_time)

            # print(" ",diff_time ," ; ",jump_frame," = ",fps)

    def getCurrentFrame(self):   
        frame,current_batch= self.buffer.buffer_frames[self.current_frame_index] 
        if self.detection_service !=None:
            if self.detection_service.get_selected_model() !=None:
                frame = self.detection_service.detect_objects(frame, 0.4)
        return frame,current_batch
        
 

            