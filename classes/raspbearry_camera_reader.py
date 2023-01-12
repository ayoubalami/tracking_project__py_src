
# from picamera2 import Picamera2, Preview
# from picamera2.encoders import H264Encoder
import json
import time
from utils_lib.utils_functions import encodeStreamingFrame  

class RaspberryCameraReader :
    def __init__(self): 
        self.stop_reading_from_user_action=True

        # self.picam2 = Picamera2()
        # self.picam2.configure(self.picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (720, 480)}))
        time.sleep(.1)

    def read_camera_stream(self):
        print("Start READING FROM raspberry camera......")
        while(True):
            image = self.picam2.capture_array()
            if self.stop_reading_from_user_action:
                time.sleep(.2)
                break
            yield from self.ProcessAndYieldFrame(image,3)

        
    def ProcessAndYieldFrame(self,frame,detection_fps):
        result={}
        copy_frame=frame.copy()
        result['detectorStream']=encodeStreamingFrame(frame=copy_frame,resize_ratio=1,jpeg_quality=80) 
        yield 'event: message\ndata: ' + json.dumps(result) + '\n\n'