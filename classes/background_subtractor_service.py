
import time
import cv2
import numpy as np

class BackgroundSubtractorService():
   
    def __init__(self):
        self.morphological_ex_iteration=3
        self.morphological_kernel_size=7
        self.blur_kernel_size=1
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(history=400,varThreshold=160,detectShadows=True)
        self.background_subtractor.setShadowValue(0)
        self.min_box_size=300
        self.deviation=0
        self.max_fps=0
        self.min_fps=1000
        self.video_resolution_ratio=1

    def reset(self):
        self.max_fps=0
        self.min_fps=1000

    def apply(self,frame,boxes_plotting=True): 

        img_height,img_width = frame.shape[:2] 
        if self.video_resolution_ratio<1:
            resized_frame=cv2.resize(frame, (int(img_width*self.video_resolution_ratio) ,int(img_height*self.video_resolution_ratio) ))
        else:
            resized_frame=frame.copy()

        start_time= time.perf_counter()

        if self.blur_kernel_size>1:
            resized_frame = cv2.GaussianBlur(resized_frame, (self.blur_kernel_size,self.blur_kernel_size), self.deviation)
        
        mask_frame = self.background_subtractor.apply(resized_frame)
        # frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if self.morphological_ex_iteration>0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morphological_kernel_size, self.morphological_kernel_size))
            mask_frame = cv2.morphologyEx(mask_frame, cv2.MORPH_CLOSE, kernel, iterations=self.morphological_ex_iteration)
            mask_frame = cv2.morphologyEx( mask_frame, cv2.MORPH_OPEN, kernel, iterations=self.morphological_ex_iteration)
            mask_frame = cv2.morphologyEx( mask_frame, cv2.MORPH_DILATE, kernel, iterations=self.morphological_ex_iteration)
        contours, _ = cv2.findContours(mask_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)        
        raw_detection_data=[]
        
        # print("__________________________________________")
        self.min_side_size=30
        # self.max_side_size=40
        relative_img_height=(img_height*self.video_resolution_ratio)
        section_count=15
        perspective_expension_ratio=0.65
        # top_margin=200
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if y<150:
                continue
            current_section=1+((y-150)*section_count)//relative_img_height
            # print("self.min_box_size*up_ratio")
            # print(img_height)
            # print((x, y, w, h))
            # print( current_section)
            # print( w*h) 
            # print( cv2.contourArea(contour))
            # print(self.min_side_size*current_section*perspective_expension_ratio)
            # print(self.max_side_size*current_section*perspective_expension_ratio)
            # print( self.video_resolution_ratio)
 
            # print("______________")
 
            if  w >self.min_side_size*current_section*perspective_expension_ratio  :
            # if  w*self.video_resolution_ratio >self.min_side_size*current_section and h*self.video_resolution_ratio>self.min_side_size*current_section:
            # if cv2.contourArea(contour) > self.min_box_size*section:
                # print(str(self.min_box_size) , str(cv2.contourArea(contour)))
                if boxes_plotting == False:
                    raw_detection_data.append((self.normalizedBBox([x, y, w, h]),1,'OBJ'))
                else:
                    #DRAW BOUNDING BOX TO MASKS
                    cv2.rectangle(mask_frame, (x, y), (x + w, y + h), 2**16-1, int(2*self.video_resolution_ratio))
                    self.drawForegroundObjectBBoxes(frame=resized_frame,xywh=(x, y, w, h))
                    self.drawForegroundObjectBBoxes(frame=frame,xywh=(x, y, w, h),readapt_bbox=True)
        if boxes_plotting:
            fps=1/round(time.perf_counter()-start_time,3)
            if fps>self.max_fps:
                self.max_fps=fps
            if fps<self.min_fps:
                self.min_fps=fps
            self.addFrameFps(resized_frame,fps)
            self.addFrameFps(frame,fps)
            return frame,resized_frame,mask_frame, fps
        else:
            return frame,raw_detection_data

    def normalizedBBox(self, xywh):
        if self.video_resolution_ratio==1:
            return xywh
        [x,y,w,h]=xywh
        x=int(x/self.video_resolution_ratio)
        y=int(y/self.video_resolution_ratio)
        w=int(w/self.video_resolution_ratio)
        h=int(h/self.video_resolution_ratio)
        return [x,y,w,h]
          
    def drawForegroundObjectBBoxes(self,frame,xywh,readapt_bbox=False):
        (x,y,w,h)=xywh
        if readapt_bbox:
            x=int(x/self.video_resolution_ratio)
            y=int(y/self.video_resolution_ratio)
            w=int(w/self.video_resolution_ratio)
            h=int(h/self.video_resolution_ratio)
        sub_img = frame[y:y+h, x:x+w]
        white_rect = np.zeros(sub_img.shape, dtype=np.uint8) 
        white_rect[:, :, 0] = 255
        res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
        frame[y:y+h, x:x+w] = res
        cv2.rectangle(frame, (x, y), (x + w, y + h),  (255, 0, 0), 2)

    def addFrameFps(self,frame,detection_fps):
        width=frame.shape[1]
        cv2.rectangle(frame,(int(width-205),10),(int(width-10),62),color=(240,240,240),thickness=-1)
        cv2.putText(frame, f'FPS: {round(detection_fps,1)}', (int(width/2)-80,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,25,50), int(2))
        cv2.putText(frame, f'MAX FPS: {round(self.max_fps,1)}', (int(width-200),31), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (25,25,250), 2)
        cv2.putText(frame, f'MIN FPS: {round(self.min_fps,1)}', (int(width-200),58), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (25,25,250), 2)
