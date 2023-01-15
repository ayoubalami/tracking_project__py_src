
import time
import cv2
import numpy as np
from utils_lib.utils_functions import addFrameFps


class BackgroundSubtractorService():
   
    def __init__(self):
        self.morphological_ex_iteration=3
        self.morphological_kernel_size=7
        self.blur_kernel_size=3
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(history=400,varThreshold=100,detectShadows=True)
        self.background_subtractor.setShadowValue(0)
        self.min_box_size=300
        self.deviation=0

    def apply(self,frame,boxes_plotting=True): 
        start_time= time.perf_counter()
        origin_frame=frame.copy()
        if self.blur_kernel_size>1:
            frame = cv2.GaussianBlur(frame, (self.blur_kernel_size,self.blur_kernel_size), self.deviation)
        frame = self.background_subtractor.apply(frame)
        # frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if self.morphological_ex_iteration>0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morphological_kernel_size, self.morphological_kernel_size))
            frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel, iterations=self.morphological_ex_iteration)
            frame = cv2.morphologyEx( frame, cv2.MORPH_OPEN, kernel, iterations=self.morphological_ex_iteration)
            frame = cv2.morphologyEx( frame, cv2.MORPH_DILATE, kernel, iterations=self.morphological_ex_iteration)
        contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)        
        raw_detection_data=[]

        for contour in contours:
            if cv2.contourArea(contour) > self.min_box_size:
                (x, y, w, h) = cv2.boundingRect(contour)
                if boxes_plotting == False:
                    raw_detection_data.append(([x, y, w, h],1,'CAR'))
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), 2**16-1, 2)
                    sub_img = origin_frame[y:y+h, x:x+w]
                    white_rect = np.zeros(sub_img.shape, dtype=np.uint8) 
                    white_rect[:, :, 0] = 255
                    res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
                    origin_frame[y:y+h, x:x+w] = res
                    cv2.rectangle(origin_frame, (x, y), (x + w, y + h),  (255, 0, 0), 2)

        if boxes_plotting:
            fps=1/round(time.perf_counter()-start_time,3)
            addFrameFps(origin_frame,fps)
            return origin_frame,frame, fps
        else:
            return origin_frame,raw_detection_data


     

