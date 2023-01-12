
import base64
import subprocess
import cv2

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


def encodeStreamingFrame(self,frame,resize_ratio=1,jpeg_quality=100):
        if resize_ratio!=1:
            frame=cv2.resize(frame, (int(self.buffer.width*resize_ratio) ,int(self.buffer.height*resize_ratio) ))
        ret,buffer=cv2.imencode('.jpg',frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        img_bytes=buffer.tobytes()
        return  base64.b64encode(img_bytes).decode()