
from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder
import time

class RaspberryCamStream :
    def __init__(self): 

        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (720, 480)}))
        time.sleep(.1)


#
#  from picamera.array import PiRGBArray
# from picamera import PiCamera
# import time

# class RaspberryCamStream :
#     def __init__(self): 
#         self.camera = PiCamera()
#         self.camera.resolution = (640, 480)
#         self.camera.framerate = 32
#         self.rawCapture = PiRGBArray(self.camera, size=(640, 480))
#         time.sleep(0.1)
   
#class RaspberryCamStream :
#    def __init__(self):
#        pass
     


   
