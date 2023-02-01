
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


def addTrackingAndDetectionTimeAndFPS(img,detection_time,tracking_time,tracking_fps):
    width=img.shape[1]
    cv2.rectangle(img,(int(width-195),10),(int(width-10),85),color=(240,240,240),thickness=-1)
    cv2.putText(img, f'FPS: {round(tracking_fps,2)}', (int(width-190),30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (25,25,250), 2)
    cv2.putText(img, f'Det. time: {round(detection_time*1000)}ms', (int(width-190),52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (250,25,25), 2)
    cv2.putText(img, f'Tra. time: {round(tracking_time*1000)}ms', (int(width-190),73), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (250,25,25), 2)
