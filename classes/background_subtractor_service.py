
import time
import cv2


class BackgroundSubtractorService():
   
    def __init__(self):
        self.backgroundSubtractor = cv2.createBackgroundSubtractorMOG2(history=50,varThreshold=200,detectShadows=True)


    def apply(self,frame):
        
        t1= time.time()
        frame = self.backgroundSubtractor.apply(frame)
        inference_time=time.time()-t1
        return frame, inference_time

