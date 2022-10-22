
import cv2,time
from buffer import Buffer
import pyshine as ps

class StreamReader: 

    def __init__(self, buffer):
        # self.target_delay=buffer.frame_duration
        self.buffer=buffer
        self.current_time=0

    def readStream(self):
        delay=self.buffer.frame_duration
        time.sleep(0.05)
        i=0
        current_sec=0
        while i< self.buffer.frames_count:
            for (frame,batch) in self.buffer.buffer_frames :
                t1= time.time()
                frame =  ps.putBText(frame,str(current_sec),text_offset_x=20,text_offset_y=20,vspace=10,hspace=10, font_scale=2.0,background_RGB=(210,220,222),text_RGB=(255,250,250))
                ret,buffer=cv2.imencode('.jpg',frame)
                img_bytes=buffer.tobytes()
                yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')
                
                i=i+1
                self.current_time=self.current_time+self.buffer.frame_duration
                new_sec=self.current_time
                # print(new_sec)
                # print(self.current_time)
                if new_sec>=current_sec+1:
                    current_sec=current_sec+1
                    new_sec=current_sec
                delay=(time.time()-t1)
                delay=self.buffer.frame_duration-delay
                if delay>0:
                    time.sleep(delay)
                # if(sec>)
                # if(round(current_time/1000/60))