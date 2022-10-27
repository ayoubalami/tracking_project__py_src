
from traceback import print_tb
import cv2,time
from buffer import Buffer
import pyshine as ps


class StreamReader: 

    def __init__(self, buffer:Buffer):
        # self.target_delay=buffer.frame_duration
        self.buffer=buffer
        self.current_time=0
        self.current_frame_index=0
        self.stop_reading_for_loading=False
        self.stop_reading_user_action=False
        self.end_of_file=False
        self.buffer.stream_reader=self

    def clean_streamer(self):
        print("STREAM_READER : begin clean_streamer")
        del(self.buffer.buffer_frames)
        self.buffer.cap.release()
        del(self.buffer.cap)
        print("STREAM_READER : end clean_streamer")


    def readStream(self):
        delay=self.buffer.frame_duration
        # delay for buffer to load some frame
        time.sleep(0.1)
        current_sec=0
        decalage=0
        while True :

            if self.stop_reading_user_action:
                time.sleep(.2)
                continue

            if (self.buffer.stop_buffring_event.is_set()):
                self.clean_streamer()
                print("STREAM_READER : end_of_thread")
                break; 

            t1= time.time()
            if self.current_frame_index>=len(self.buffer.buffer_frames) or len(self.buffer.buffer_frames)==0 :
                continue
            frame,current_batch=self.buffer.buffer_frames[self.current_frame_index]            
            frame =  ps.putBText(frame,str(current_sec),text_offset_x=20,text_offset_y=20,vspace=10,hspace=10, font_scale=2.0,background_RGB=(210,220,222),text_RGB=(255,250,250))
            ret,buffer=cv2.imencode('.jpg',frame)
            img_bytes=buffer.tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')
            self.current_time=self.current_time+self.buffer.frame_duration
            new_sec=self.current_time
            if new_sec>=current_sec+1:
                current_sec=current_sec+1
                new_sec=current_sec
            
                
            # print("***************")
            # print(" TOTAL FRAMES IN MEMORY : "+str(len(self.buffer.buffer_frames)))
            # print("current_frame_index : "+str(self.current_frame_index))
            # print("current_batch : "+ str(current_batch))
            # print("last_batch : "+ str(self.buffer.last_batch))
            # print("last_frame_of_last_batch : "+ str(self.buffer.last_frame_of_last_batch))
            # print("==============")

            self.current_frame_index=self.current_frame_index+1
            # DELETE PREVIOUS BATCH FRAMES ALREADY PRINTED FROM MEMORY
            if (self.current_frame_index==self.buffer.batch_size):
                self.buffer.delete_last_batch(current_batch)
                self.current_frame_index=0
                self.buffer.download_new_batch=True
                # print("<<<<<< decalage >>>>" + str(decalage))
            
            # END READING IN CASE OF REACHING THE LAST BATCH
            if ( self.buffer.last_frame_of_last_batch==self.current_frame_index  and current_batch==self.buffer.last_batch):
                self.clean_streamer()
                print("STREAM_READER : end_of_file")
                break;   

            # CALCULATE TIME OF INSTRUCTIONS AND ADD DELAY TO REACH THE FRAME DURATION 
            calculeTime=(time.time()-t1)

            # print(">>> self.buffer.frame_duration-calculeTime :" +str(self.buffer.frame_duration-calculeTime))
            # print(">>> decalage :" +str(decalage))
            # print(">>> frame_duration :" +str(self.buffer.frame_duration))

            delay=self.buffer.frame_duration-calculeTime + decalage
            if delay>0:
                decalage=0
                time.sleep(delay)
            else:
                if delay<0 :
                    decalage=delay
            # time.sleep(0.05)
            # print("LOOOOP" + str(self.current_frame_index))
        
        
         
         
           
           
