
from math import floor
import threading
import cv2,time
import pafy
class Buffer:

    cap=None

    def reset(self, video_src=None,youtube_url=None):
        if self.cap != None:
            self.cap.release()

        if youtube_url != None :
            print("====>>>load_from_youtube")
            self.cap=self.load_from_youtube(youtube_url)
            self.init_params()
            return

        if video_src != None:
            print("====>>>load_from_local_video")
            self.cap=self.load_from_local_video(video_src)
            self.init_params()
            return

    def init_params(self):
        self.buffer_frames = []
        self.batch_size=50 # 200 frame 
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        self.frames_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_duration= 1/self.fps 
        self.video_duration = self.frames_count/ self.fps
        # self.last_loaded_frame_index=0
        self.stream_reader=None
        self.current_batch=0
        self.last_batch=floor(self.frames_count/self.batch_size)
        self.last_frame_of_last_batch=floor(self.frames_count%self.batch_size)
        self.download_new_batch=True
        self.stop_buffring_event=threading.Event() 
       

    def __init__(self, video_src=None,youtube_url=None): 
        self.reset(video_src=video_src,youtube_url=youtube_url)

    def __del__(self):
        print("DELETE DESTRUCTER")
        if  self.cap != None:
            self.cap.release()
        if  self.buffer_frames != None:
            del self.buffer_frames

    def delete_last_batch(self,to_delete_batch):
        del self.buffer_frames[0:self.batch_size]
        # print("delete last batch : "+str(to_delete_batch))

    def downloadBuffer(self):
        print("Start CALL downloadBuffer")
        while True:
            if self.stop_buffring_event.is_set():
                print("BUFFER : THREAD IS SET ")
                # self.stop_buffring_event
                print("BUFFER : buffer_frames SUCCESSIFLY DELETED ")   
                break
            try:
                if(self.download_new_batch or self.current_batch<2):
                    # print("====> loading frames ! for batch :"+ str(self.current_batch ))
                    for i in range(self.batch_size):
                        success, frame = self.cap.read()
                        if success == False :
                            # print(" ==== END STREAM ")
                            return
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


    def load_from_youtube(self,youtube_url:str):
        urlPafy = pafy.new(youtube_url)
        videoplay = urlPafy.getbest(preftype="any")
        streams = urlPafy.allstreams
        videoplay = urlPafy.getbestvideo(preftype="mp4")
        resolution_by_priority=[480,360,720]

        for res in resolution_by_priority:
            indexes = [i for i in range(len(streams)) if res == streams[i]._dimensions[1] and streams[i]._mediatype == 'video' ]
            if len(indexes)>0:
                videoplay = streams[indexes[0]]
                break       
        return cv2.VideoCapture(videoplay.url)

    def load_from_local_video(self,video_src:str):
        return cv2.VideoCapture(video_src)

    def clean_memory(self):
        print("clean memory ====")
        # del self.buffer_frames

        