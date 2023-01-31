
import base64
from csv import writer
import subprocess
import cv2
from utils_lib.enums import ClientStreamTypeEnum
import numpy as np
def runcmd(cmd, verbose = False, *args, **kwargs):
    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass

def addFrameFps(img,detection_fps):
    width=img.shape[1]
    cv2.putText(img, f'FPS: {round(detection_fps,2)}', (int(width/2)-20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,25,50), 2)
    
def addTrackingFrameFps(img,detection_fps):
    width=img.shape[1]
    if(detection_fps!=None):
        cv2.putText(img, f'Tracking FPS: {round(detection_fps,2)}', (int(width/2)-100,50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (50,25,255), 2)
    else:
        cv2.putText(img, f'Tracking FPS: ++', (int(width/2)-100,50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (50,25,255), 2)

def addTrackingAndDetectionTime(img,detection_time,tracking_time):
    width=img.shape[1]
    cv2.rectangle(img,(int(width-195),10),(int(width-10),70),color=(240,240,240),thickness=-1)
    cv2.putText(img, f'Det. time: {round(detection_time*1000)}ms', (int(width-190),30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (250,25,25), 2)
    cv2.putText(img, f'Tra. time: {round(tracking_time*1000)}ms', (int(width-190),60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (250,25,25), 2)
