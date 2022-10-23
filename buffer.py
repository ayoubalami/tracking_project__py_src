
from math import floor
import cv2,time
import pafy

class Buffer:

    def __init__(self, video_src):
        self.video_src = video_src
        # urlPafy = pafy.new(video_src)
        # videoplay = urlPafy.getbest(preftype="any")
        # streams = urlPafy.allstreams
        # videoplay = urlPafy.getbestvideo(preftype="mp4")
        # resolution_by_priority=[480,360,720]

        # for res in resolution_by_priority:
        #     indexes = [i for i in range(len(streams)) if res == streams[i]._dimensions[1] and streams[i]._mediatype == 'video' ]
        #     if len(indexes)>0:
        #         videoplay = streams[indexes[0]]
        #         break       
        # self.cap = cv2.VideoCapture(videoplay.url)

        self.buffer_frames = []
        self.batch_size=75 # 200 frame 
        self.cap = cv2.VideoCapture(self.video_src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        self.frames_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_duration= 1/self.fps 
        self.video_duration = self.frames_count/ self.fps
        self.last_loaded_frame_index=0
        self.stream_reader=None
        self.current_batch=0
        self.last_batch=floor(self.frames_count/self.batch_size)
        self.last_frame_of_last_batch=floor(self.frames_count%self.batch_size)
        self.download_new_batch=True

    def delete_last_batch(self,to_delete_batch):
        del self.buffer_frames[0:self.batch_size]
        print("delete last batch : "+str(to_delete_batch))
         

    def downloadBuffer(self):
         # self.buffer_frames = []
        while True:
            try:
                if(self.download_new_batch or self.current_batch<2):
                    print("====> loading frames ! for batch :"+ str(self.current_batch ))
                    for i in range(self.batch_size):
                        success, frame = self.cap.read()
                        if success == False :
                            print(" ==== END STREAM ")
                            return
                        self.buffer_frames.append((frame,self.current_batch))
                        self.last_loaded_frame_index=self.last_loaded_frame_index+1
                        # self.cap.release()

                    # print("===> End loading frames !")
                    self.download_new_batch=False
                    print(len(self.buffer_frames))
                    print(self.current_batch)
                    
                    # self.delete_last_batch()
                    # print("AFTER DELETE ====")
                    # print(len(self.buffer_frames))

                    if self.stream_reader!=None :
                        if self.stream_reader.stop_reading_for_loading==True :
                            self.stream_reader.stop_reading_for_loading=False
                            

                    self.current_batch=self.current_batch+1
                else:
                    time.sleep(0.1)
            except:
                print(" ERROR ")
                break

        # return self.buffer_frames