
import cv2,time
from buffer import Buffer
import pyshine as ps


class StreamReader: 

    def __init__(self, buffer:Buffer):
        # self.target_delay=buffer.frame_duration
        self.buffer=buffer
        self.current_time=0
        self.last_readed_frame_index=0
        self.stop_reading_for_loading=False
        self.end_of_file=False
        self.buffer.stream_reader=self
        self.read_from=0

    def readStream(self):
        delay=self.buffer.frame_duration
        time.sleep(0.05)
        i=0
        current_sec=0
        
        while True :
            
            if self.stop_reading_for_loading or len(self.buffer.buffer_frames)==0 :
                continue
            frame,current_batch=self.buffer.buffer_frames[self.last_readed_frame_index]

            t1= time.time()
            frame =  ps.putBText(frame,str(current_sec),text_offset_x=20,text_offset_y=20,vspace=10,hspace=10, font_scale=2.0,background_RGB=(210,220,222),text_RGB=(255,250,250))
            ret,buffer=cv2.imencode('.jpg',frame)
            img_bytes=buffer.tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')
            self.current_time=self.current_time+self.buffer.frame_duration
            new_sec=self.current_time
            
            if new_sec>=current_sec+1:
                current_sec=current_sec+1
                new_sec=current_sec
        
            self.last_readed_frame_index=self.last_readed_frame_index+1
            
            if(self.last_readed_frame_index==self.buffer.last_loaded_frame_index):
                self.stop_reading_for_loading=True
                self.read_from=self.read_from+self.buffer.batch_size
                
            # print(" TOTAL FRAMES IN MEMORY : "+str(len(self.buffer.buffer_frames)))
            # print("last_readed_frame_index : "+str(self.last_readed_frame_index))
            # print("last_loaded_frame_index : "+str(self.buffer.last_loaded_frame_index))
            # print("read_from : "+ str(self.read_from))
            # print("current_batch : "+ str(current_batch))
            # print("last_batch : "+ str(self.buffer.last_batch))
            # print("last_frame_of_last_batch : "+ str(self.buffer.last_frame_of_last_batch))
            # print("==============")

            # DELETE PREVIOUS BATCH FRAMES ALREADY PRINTED FROM MEMORY
            if (self.last_readed_frame_index==self.buffer.batch_size):
                self.buffer.delete_last_batch(current_batch)
                self.last_readed_frame_index=0
                self.buffer.download_new_batch=True
            
            
            # END READING IN CASE OF REACHING THE LAST BATCH
            if ( self.buffer.last_frame_of_last_batch==self.last_readed_frame_index  and current_batch==self.buffer.last_batch):
                    # self.end_of_file=True 
                    break;   


            # CALCULATE TIME OF INSTRUCTIONS AND ADD DELAY TO REACH THE FRAME DURATION 
            calculeTime=(time.time()-t1)
            delay=self.buffer.frame_duration-calculeTime
            
            if delay>0:
                time.sleep(delay)

            # time.sleep(0.05)
            # print("LOOOOP" + str(self.last_readed_frame_index))
        del(self.buffer.buffer_frames)
        self.buffer.cap.release()
        del(self.buffer.cap)
        print("end_of_file")
            # print("global_frame_index : "+str(global_frame_index))
            # print("self.buffer.buffer_frames : "+str(len(self.buffer.buffer_frames)))
            # print("last_readed_frame_index : "+str(self.last_readed_frame_index))
            # print("last_loaded_frame_index : "+str(self.buffer.last_loaded_frame_index))
            # print("stop_reading : "+str(self.stop_reading))
            # print("read_from : "+ str(self.read_from))
            # print("==============")
           
           
