# from turtle import shape
from concurrent.futures import ThreadPoolExecutor
import math
from time import sleep
from buffer import Buffer
from stream_reader import StreamReader

 
def generate_frames_from_youtube_with_cache():
    # https://www.youtube.com/watch?v=luxaGPxBVPo
    # video_src='https://www.youtube.com/watch?v=KBsqQez-O4w'
    video_src = "highway1.mp4"
    buffer=Buffer(video_src)
    with ThreadPoolExecutor() as executor:
        future2 = executor.submit(buffer.downloadBuffer())
        future1 = executor.submit(read(buffer))

        # future2.result()
        # future1.result()

        # print(future.result())
        # future.result()
     
    # stream_reader=StreamReader(buffer)
    # yield from stream_reader.readStream() 
 
 
def read(buffer):
    with open("test.txt", 'w') as f:
        for i in range(len(buffer.buffer_frames)):
            # print(i)
            (frame,batch)=buffer.buffer_frames[i]
            f.write(str(i)+" "+str(batch))
            f.write("\n")
            sleep(0.03333333)
            
generate_frames_from_youtube_with_cache()