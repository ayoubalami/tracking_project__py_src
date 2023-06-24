from warnings import catch_warnings
import cv2
from threading import Condition,Thread

class WebcamReader:
    def __init__(self, src=0):

            self.stopped = False
            if src=='RASPBERRY_CAM':
                self.stream=cv2.VideoCapture(1,  cv2.CAP_V4L)
            else:
                self.stream = cv2.VideoCapture(src)
            if not self.stream.isOpened():
                raise Exception("Couldn't open camera {}".format(src))
            self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            (self.grabbed, self.frame) = self.stream.read()
            self.hasNew = self.grabbed
            self.condition = Condition()
            Thread(target=self.update, args=()).start()
            print("DONE==")
       
    def start(self):
        self.stopped=False

    def update(self,):
        while True:
            if self.stopped: return
            (self.grabbed, self.frame) = self.stream.read()
            with self.condition:
                self.hasNew = True
                self.condition.notify_all()
            
    def read(self):
        try:
            if not self.hasNew:
                with self.condition:
                    self.condition.wait()

            self.hasNew = False
            return True,self.frame
        except:
            return False,None

    def stop(self):
        self.stopped = True


   
