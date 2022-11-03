from warnings import catch_warnings
import cv2
from threading import Condition,Thread

class WebcamStream:
    def __init__(self, src=0):
        self.stopped = False

        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        # self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
        # self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 600)

        (self.grabbed, self.frame) = self.stream.read()
    
        self.hasNew = self.grabbed
        self.condition = Condition()

    def start(self):

        Thread(target=self.update, args=()).start()
        return self

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


   