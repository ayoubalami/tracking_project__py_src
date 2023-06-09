
from math import floor
import threading
import cv2,time
from utils_lib.enums import StreamSourceEnum
class Buffer:

    cap=None
    stream_reader=None
    video_start_seconde=0

    def init_params(self,starting_second=0):
        self.buffer_frames = []
        self.batch_size=50 # 200 frame 
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frames_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width  = self.cap.get(3)
        self.height = self.cap.get(4)
        print("VIDEO DIMENSION : " +str(self.width) +" x "+ str(self.height))

        # starting_second=60
        self.set_buffer_starting_frame(starting_second)
        
        self.frame_duration= 1/self.fps 
        self.video_duration = self.frames_count/ self.fps
        # self.last_loaded_frame_index=0
        self.current_batch=0
        self.last_batch=floor(self.frames_count/self.batch_size)
        self.last_frame_of_last_batch=floor(self.frames_count%self.batch_size)
        self.download_new_batch=True
        self.stop_buffring_event=threading.Event() 

    def __init__(self, stream_source : StreamSourceEnum, video_src,stream_reader): 
        self.video_play=None
        self.stream_reader=stream_reader
        if stream_source==StreamSourceEnum.FILE:
            self.reset_file_buffer(file_src=video_src)
            self.batch_size=50
 

    def reset_file_buffer(self, file_src=None,starting_second=0):
        if self.cap != None:
            self.cap.release()
        if file_src != None:
            print("====>>>RESET LOCAL FILE ")
            print(self.cap)
            self.cap=self.load_from_local_video(file_src)
            self.init_params(starting_second=starting_second)
 
    def delete_last_batch(self,to_delete_batch):
        if self.buffer_frames:
            del self.buffer_frames[0:self.batch_size]

    def download_buffer(self):
        print("Start CALL downloadBuffer")
        while True:
            if self.stop_buffring_event.is_set():
                print("BUFFER : THREAD IS SET ")
                break
            try:
                # print("download_buffer loop")
                if(self.download_new_batch or self.current_batch<2):
                    # print("====> loading frames ! for batch :"+ str(self.current_batch ))
                    for i in range(self.batch_size):
                        success, frame = self.cap.read()
                        if success == False :
                            # print(" |||| success == False ")
                            time.sleep(0.1)
                            continue
                        self.buffer_frames.append((frame,self.current_batch))
                        # self.last_loaded_frame_index=self.last_loaded_frame_index+1

                    self.download_new_batch=False
                    if self.stream_reader!=None :
                        if self.stream_reader.stop_reading_for_loading==True :
                            self.stream_reader.stop_reading_for_loading=False
                            
                    self.current_batch=self.current_batch+1
                    # time.sleep(5)
                else:
                    time.sleep(0.1)
            except:
                print(" ERROR ")
                break
        print(":::: END downloadBuffer LOOP")

         

    def load_from_local_video(self,video_src:str):
        return cv2.VideoCapture(video_src)

    def clean_memory(self):
        print("CALL DESTRUCTER FROM BUFFER")
        self.stop_buffring_event.set()
        if  self.cap != None:
            self.cap.release()
            del self.cap
        if  self.buffer_frames != None:
            del self.buffer_frames


    def  set_buffer_starting_frame(self,starting_second):

        # print(starting_second)
        # print(self.frames_count/self.fps)

        starting_second=(int)((starting_second*(self.frames_count/self.fps))/100)

        print("=====")
        print(starting_second)

        self.video_start_second=starting_second
        frame_position = self.video_start_second*  self.fps # Replace with the desired frame number
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
        print(f"START VIDEO FROM FRAME NUMBER : {frame_position} , video_start_second : {self.video_start_second} ")
        # if self.stream_reader:
        self.stream_reader.current_sec= self.video_start_second
        self.stream_reader.current_time= self.video_start_second

