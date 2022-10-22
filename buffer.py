
from ast import Break
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
        self.batch_size=200 # 200 frame 
        self.cap = cv2.VideoCapture(self.video_src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        self.frames_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_duration= 1/self.fps 
        self.video_duration = self.frames_count/ self.fps
        self.last_loaded_frame_index=0
        self.stream_reader=None

        # print("str(self.fps)")
        # print(str(self.fps))
        # print(str(self.frames_count))
        # print(str(self.frame_duration))

    def downloadBuffer(self):
        batch=0
        # self.buffer_frames = []
        while True:
            try:
                
                print("Start loading frames !")
                for i in range(self.batch_size):
                    success, frame = self.cap.read()
                    if success == False :
                        print(" ==== END STREAM ")
                        return
                    self.buffer_frames.append((frame,batch))
                    self.last_loaded_frame_index=self.last_loaded_frame_index+1
                    # self.cap.release()
                print("End loading frames !")
                print(len(self.buffer_frames))
                if self.stream_reader!=None :
                    if self.stream_reader.stop_reading_for_loading==True :
                        self.stream_reader.stop_reading_for_loading=False
                        

                batch=batch+1
                # time.sleep(0.)
            except:
                print(" ERROR ")
                break

        # return self.buffer_frames