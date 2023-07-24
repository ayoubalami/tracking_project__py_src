
import base64
import json
from math import floor
from queue import Queue
import threading
import cv2,time
from utils_lib.enums import StreamSourceEnum

class VideoStream:
    video_file= "highway2.mp4"
    frames_Q = Queue(maxsize=128)
    interrepted_stream=True
    activate_real_time_stream_simulation=True
    starting_second=0
    stop_cv2reader=False

    def init_params(self):
        self.stopped = True
       
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frames_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width,self.height  = self.cap.get(3),self.cap.get(4)
        print("VIDEO DIMENSION : " +str(self.width) +" x "+ str(self.height))        
        self.frame_duration= 1/self.fps 
        self.video_duration = self.frames_count/ self.fps
        self.current_video_time=0
        self.first_frame=[]
        self.one_next_frame=True
        self.frames_to_jump=0
        self.remain_after_frames_jump=0

    def __init__(self,stream_source:StreamSourceEnum): 
        if stream_source==StreamSourceEnum.FILE:
            self.cap=cv2.VideoCapture("videos/"+self.video_file)   
        elif stream_source==StreamSourceEnum.WEBCAM:
            self.cap=cv2.VideoCapture("http://10.10.23.14:9000/video")

        self.init_params()
        self.start_thread()

    def get_frames(self):  
        self.start_time=time.perf_counter() 
        while self.has_next() :
            if not self.stopped or self.one_next_frame:
                frame = self.get_next()
                yield frame 
                if self.activate_real_time_stream_simulation:
                    self.synchronize_frame()
                self.one_next_frame=False
            else:
                # Wait if the stream is stopped or no frame is there
                time.sleep(0.1)
                self.interrepted_stream=True

    def init_start_time(self):
        self.frames_to_jump=0
        self.remain_after_frames_jump=0
        self.start_time=time.perf_counter() 

    def synchronize_frame(self):
        def delay_to_keep_fps(diff_duration):
            time.sleep(diff_duration)
            self.frames_to_jump=0
            self.remain_after_frames_jump=0

        def jump_frames(retard):
            if self.interrepted_stream :
                self.interrepted_stream=False
                return
            self.frames_to_jump=(int)((retard+self.remain_after_frames_jump)//self.frame_duration)
            self.remain_after_frames_jump=(retard+self.remain_after_frames_jump)%self.frame_duration

        current_duration=(time.perf_counter()-self.start_time) 
        diff_duration=self.frame_duration-current_duration
        if diff_duration>0:
            delay_to_keep_fps(diff_duration)
        else:
            jump_frames(-diff_duration)
        self.start_time=time.perf_counter()


    def start_thread(self):
        self.t = threading.Thread(target=self.update, args=())
        self.t.daemon = True
        self.t.start()
        return self

    def update(self):
        while True:            
            if not self.frames_Q.full() and not self.stop_cv2reader:
                (grabbed, frame) = self.cap.read()
                # print(self.frames_Q.qsize())
                if not grabbed :
                    print(" no frame to grab...")
                    self.stop()
                    time.sleep(0.01)
                self.frames_Q.put(frame)
            
            if len(self.first_frame)==0 and not self.stop_cv2reader:
                (grabbed, frame) = self.cap.read()
                if grabbed :
                    print("set FIRST FRAME")
                    self.first_frame=frame

            # read from video quickly if its size is under 4 
            if self.frames_Q.qsize()<4:
                time.sleep(0.003)
            else:
                time.sleep(0.03)

    def stop(self):
        self.stopped = True

    def start(self):
        self.stopped = False

    def has_next(self):
        return self.frames_Q.qsize() > 0 or self.stopped

    def get_next(self):
        if self.frames_to_jump>0:
            for _ in range(self.frames_to_jump):
                self.current_video_time+=self.frame_duration
                self.frames_Q.get()

        self.current_video_time+=self.frame_duration
        return self.frames_Q.get()

    def reset(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.stopped = True
        new_queue=Queue(maxsize=128)
        new_queue.put(self.first_frame)
        self.frames_Q=new_queue
        # self.capture_first_frame=True
        self.current_video_time=0
        self.one_next_frame=True

    def set_starting_second(self,second):
        self.stop_cv2reader=True
        # self.stopped = True
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, second*self.fps)
        new_queue=Queue(maxsize=128)
        (grabbed, frame) = self.cap.read()
        if grabbed:
            new_queue.put(frame)
        self.frames_Q=new_queue
        self.current_video_time=second
        self.one_next_frame=True
        self.stop_cv2reader=False


    def change_video_file(self,video_file):
        if self.video_file!=video_file:
            self.video_file=video_file
            self.init_params()
            # wait for update() to load a self.first_frame to put in frames_Q
            # while(len(self.first_frame)==0):
            #     time.sleep(.05)
            #     print("wait for first frame")
            self.clean_frames_Q()
            self.frames_Q.put(self.first_frame)
            self.one_next_frame=True

    def clean_frames_Q(self):
        for _ in range(self.frames_Q.qsize()):
            self.frames_Q.get()
