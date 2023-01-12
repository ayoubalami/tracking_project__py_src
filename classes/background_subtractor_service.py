
import time
import cv2
import numpy as np


class BackgroundSubtractorService():
   
    def __init__(self):
        self.morphological_ex_iteration=0
        self.kernel_size=5
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.min_box_size=1000

    def apply(self,frame): 
        start_time= time.perf_counter()
        origin_frame=frame.copy()
        frame = self.background_subtractor.apply(frame)
        # frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        kernel = np.ones((self.kernel_size,self.kernel_size), np.uint8)
        frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel, iterations=self.morphological_ex_iteration)
        contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > self.min_box_size:
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), 2**16-1, 2)

                sub_img = origin_frame[y:y+h, x:x+w]
                white_rect = np.zeros(sub_img.shape, dtype=np.uint8) 
                white_rect[:, :, 0] = 255

                res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
                origin_frame[y:y+h, x:x+w] = res
                cv2.rectangle(origin_frame, (x, y), (x + w, y + h),  (255, 0, 0), 2)

        fps=1/round(time.perf_counter()-start_time,3)
        self.addFrameFps(origin_frame,fps)
        return origin_frame,frame, fps

    def addFrameFps(self,img,detection_fps):
        width=img.shape[1]
        cv2.putText(img, f'FPS: {round(detection_fps,2)}', (int(width/2)-20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,25,50), 2)
    

