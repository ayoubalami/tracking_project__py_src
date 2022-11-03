
from ast import Str
from curses import beep
from math import floor
import threading
import cv2,time,os, numpy as np
from classes.buffer import Buffer
import pyshine as ps
from classes.detection_service import IDetectionService
from  classes.WebcamStream import WebcamStream
import tensorflow as tf

class StreamReader: 

    np.random.seed(123)
    # threshold = 0.5
    perf = []
    classAllowed=[]
    colorList=[]
    buffer :Buffer= None
    def __init__(self, detection_service:IDetectionService,video_src=None,youtube_url=None, camera_src=None):
       
        self.stop_reading_from_user_action=True
        self.current_time=0
        self.detection_service=detection_service

        if camera_src !=None: 
            self.camera_src=camera_src
            self.webcam_stream = WebcamStream(src=self.camera_src).start()
            return 

        # self.buffer=Buffer(youtube_url=self.youtube_url)
        self.buffer=Buffer(video_src=video_src,youtube_url=youtube_url)
   
        self.current_frame_index=0
        self.stop_reading_for_loading=False
        self.end_of_file=False
        self.buffer.stream_reader=self
        self.video_src=video_src
        self.youtube_url=youtube_url

        self.buffering_thread= threading.Thread(target=self.buffer.download_buffer)
        self.buffering_thread.start()
        print("StreamReader created .....")
        
    def clean_streamer(self):
        print("================== >>STREAM_READER : begin clean_streamer")
        del(self.buffer.buffer_frames)
        self.buffer.cap.release()
        del(self.buffer.cap)
        print("STREAM_READER : end clean_streamer")


    def reset(self):
        self.buffer.reset(video_src=self.video_src,youtube_url=self.youtube_url)             
        self.current_time=0
        self.current_sec=0
        self.current_frame_index=0


    def read_stream(self):
        if self.camera_src!=None:   
            yield from self.read_stream_from_webcam()
        else:
            yield from self.read_stream_from_buffer()


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
                    frame = self.detection_service.detect_objects(frame, 0.4)
            
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
            # print("Start LOOPING READING ......")
            t1= time.time()
            # IF BUTTON STOP PRESSED CONTINUE
            if self.stop_reading_from_user_action:
                time.sleep(.2)
                continue
            
             # END READING IN CASE OF REACHING THE LAST BATCH
            if ( self.buffer.last_frame_of_last_batch==self.current_frame_index  and current_batch==self.buffer.last_batch):
                # self.clean_streamer()
                # print("STREAM_READER : end_of_file")
                continue;   

            if (self.buffer.stop_buffring_event.is_set()):
                self.clean_streamer()
                print("STREAM_READER : end_of_thread")
                # time.sleep(.2)
                # self.buffer.stop_buffring_event.clear()
                break; 

            if self.current_frame_index>=len(self.buffer.buffer_frames) or len(self.buffer.buffer_frames)==0 :
                continue
            
            # SENT FRAMES TO NAV
            frame,current_batch=self.getCurrentFrame() 
            
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
            current_fps=round(1/diff_time)

            # print(" ",diff_time ," ; ",jump_frame," = ",fps)

    def getCurrentFrame(self):   
        frame,current_batch= self.buffer.buffer_frames[self.current_frame_index] 
        if self.detection_service !=None:
            if self.detection_service.get_selected_model() !=None:
                frame = self.detection_service.detect_objects(frame, 0.4)
        return frame,current_batch
        
 

            