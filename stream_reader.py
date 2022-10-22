
import cv2,time
from buffer import Buffer
import pyshine as ps


class StreamReader: 

    def __init__(self, buffer:Buffer):
        # self.target_delay=buffer.frame_duration
        self.buffer=buffer
        self.current_time=0
        self.last_readed_frame_index=0
        self.stop_reading=False
        self.buffer.stream_reader=self
        self.read_from=0

    def readStream(self):
        delay=self.buffer.frame_duration
        time.sleep(0.05)
        i=0
        current_sec=0
        
        while i< self.buffer.frames_count :
            for  global_frame_index,(frame,batch) in enumerate(self.buffer.buffer_frames[self.read_from:]) :
                # print("self.buffer.last_loaded_frame_index")
                # print(self.buffer.last_loaded_frame_index)
                # input("Press Enter to continue...")
                if self.stop_reading :
                    continue

                # print("++++++ read_from ++++++")
                # print(self.read_from)
                # print("======="+ str(global_frame_index)+"+ =======")
                # print(global_frame_index)

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
            
            
                self.last_readed_frame_index=self.last_readed_frame_index+1
                
                if(self.last_readed_frame_index==self.buffer.last_loaded_frame_index):
                    self.stop_reading=True
                    self.read_from=self.read_from+self.buffer.batch_size
               
                delay=(time.time()-t1)
                delay=self.buffer.frame_duration-delay
                
                if delay>0:
                    time.sleep(delay)

                
                print("global_frame_index : "+str(global_frame_index))
                print("self.buffer.buffer_frames : "+str(len(self.buffer.buffer_frames)))
                print("last_readed_frame_index : "+str(self.last_readed_frame_index))
                print("last_loaded_frame_index : "+str(self.buffer.last_loaded_frame_index))
                print("stop_reading : "+str(self.stop_reading))
                print("read_from : "+ str(self.read_from))
                print("==============")
                # print("<<<<DELAY>>>>")
                # print(delay)

            # self.read_from=self.buffer.last_loaded_frame_index
            print("<<<< New featching >>>>")
            time.sleep(0.1)
            # print( self.read_from)
            # print("<<<<S==============>>>>")
