
import base64
import json
from math import floor
from queue import Queue
import threading
import cv2,time
from utils_lib.enums import StreamSourceEnum
from classes.detection_services.detection_service import IDetectionService

class VideoStream:
    cap=None
    stream_reader=None
    video_start_second=0
    video_file= "highway2.mp4"
    current_video_time=0
    frames_Q = Queue(maxsize=128)
    stopped = True
    one_next_frame=True
    first_frame=[]
    detection_service: IDetectionService
    def init_params(self):
        self.stopped = True
        self.cap=cv2.VideoCapture("videos/"+self.video_file)
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

    def __init__(self): 
        self.init_params()
        self.start_thread()

    def get_frames(self):  
        self.start_time=time.perf_counter() 
        frame=self.get_next()
        while self.has_next() :
            # print("get next")
            if not self.stopped or self.one_next_frame:
                frame = self.get_next()
                yield frame 
                self.delay_to_keep_fps()
                self.one_next_frame=False


    def delay_to_keep_fps(self):
        current_duration=(time.perf_counter()-self.start_time)
        if self.frame_duration-current_duration>0:
            time.sleep((self.frame_duration-current_duration))
        self.start_time=time.perf_counter()
        self.current_video_time+=self.frame_duration

    def start_thread(self):
        self.t = threading.Thread(target=self.update, args=())
        self.t.daemon = True
        self.t.start()
        return self

    def update(self):
        while True:            
            if not self.frames_Q.full():
                (grabbed, frame) = self.cap.read()
                if not grabbed :
                    print("stop...")
                    self.stop()
                    time.sleep(0.01)
                self.frames_Q.put(frame)
            
            if len(self.first_frame)==0:
                (grabbed, frame) = self.cap.read()
                if grabbed :
                    print("set FIRST FRAME")
                    self.first_frame=frame

            if self.frames_Q.qsize()>3:
                time.sleep(0.03)
            else:
                time.sleep(0.001)

    def stop(self):
        self.stopped = True

    def start(self):
        self.stopped = False

    def has_next(self):
        return self.frames_Q.qsize() > 0

    def get_next(self):
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

    def change_video_file(self,video_file):
        if self.video_file!=video_file:
            self.video_file=video_file
            self.init_params()
            # wait for update() to load a self.first_frame to put in frames_Q
            while(len(self.first_frame)==0):
                time.sleep(.05)
                print("wait for first frame")
            self.clean_frames_Q()
            self.frames_Q.put(self.first_frame)
            self.one_next_frame=True

    def clean_frames_Q(self):
        for _ in range(self.frames_Q.qsize()):
            self.frames_Q.get()
