
import base64
import subprocess
import cv2
from utils_lib.enums import ClientStreamTypeEnum

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
    cv2.putText(img, f'Tracking FPS: {round(detection_fps,2)}', (int(width/2)-100,50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (50,25,255), 2)

 
