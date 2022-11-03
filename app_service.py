# from turtle import shape
import threading,cv2
from time import sleep
from unittest import result
from flask import jsonify,stream_with_context,Flask,render_template,Response
from classes.buffer import Buffer
from classes.tensorflow_detection_service import TensorflowDetectionService
from classes.stream_reader import StreamReader
from classes.detection_service import IDetectionService
from classes.WebcamStream import WebcamStream


class AppService:
    
    stream_reader :StreamReader = None
 
    detection_service :IDetectionService= None

    # video_src = "videos/highway1.mp4"
    youtube_url=None
    camera_src='http://10.10.23.223:9000/video'
    video_src = None


    # youtube_url = "https://www.youtube.com/watch?v=nt3D26lrkho"

    # youtube_url = "https://www.youtube.com/watch?v=QuUxHIVUoaY"
    # youtube_url = "https://www.youtube.com/watch?v=nV2aXhxoJ0Y"
    # youtube_url = "https://www.youtube.com/watch?v=TW3EH4cnFZo"
    # youtube_url = "https://www.youtube.com/watch?v=7y2oOsucOdc"
    # youtube_url = "https://www.youtube.com/watch?v=nt3D26lrkho"
    # youtube_url = "https://www.youtube.com/watch?v=KBsqQez-O4w"
   

    buffering_thread=None

    def __init__(self):
        print("AppService Starting ...")

        self.detection_service=TensorflowDetectionService()
       
        if self.detection_service!=None :
            print( " detection_module loaded succesufuly")
            print( "Service name : ",self.detection_service.service_name())
        else :
            print( " No detection_module To load")

        print("AppService Started.")

        
    def reset_stream(self):
        self.stream_reader.reset()
        return jsonify('reset stream')
  

    def index(self):
        return render_template('index.html')
 
   

    def return_stream(self):
        yield from self.stream_reader.read_stream()
        # yield from self.webcam_stream.read_from_camera()

    def video_stream(self):
        self.stream_reader=StreamReader(self.detection_service, video_src=self.video_src,youtube_url=self.youtube_url,camera_src=self.camera_src)  
        return Response(self.return_stream(),mimetype='multipart/x-mixed-replace; boundary=frame')

      
  
    # def timer(self):
    #     if self.stream_reader != None :
    #         return jsonify(result=self.stream_reader.current_time)
    #     else :
    #         return jsonify(result=1)

    # def video_duration(self):
    #     return jsonify(result=self.buffer.video_duration)

    def clean_memory(self):
        print(" ||||||| clean_memory ")
        # if self.buffer :
        # # print(" ||||||| ENTER ")
        #     self.buffer.stop_buffring_event.set()
        return jsonify(result='DONE')
        return jsonify(result='clean_memory ERROR')

    def stop_stream(self):
        if (self.stream_reader.buffer or self.stream_reader.webcam_stream ) and not self.stream_reader.stop_reading_from_user_action :
            self.stream_reader.stop_reading_from_user_action=True
            return jsonify(result='stream stoped')
        return jsonify(result='error server in stream stoped')

    def start_stream(self):
        # wait for streamer to be created before starting
        while(True):
            sleep(0.01)
            if self.stream_reader :
                if (self.stream_reader.buffer or self.stream_reader.webcam_stream ) and self.stream_reader.stop_reading_from_user_action :
                    self.stream_reader.stop_reading_from_user_action=False
                    return jsonify(result='stream started')
            # return jsonify(result='error server in stream started')

    
    def get_object_detection_list(self):
        if self.detection_service!=None :
            return jsonify(self.detection_service.get_object_detection_models())
      
    def load_detection_model(self,model=None):
        if self.detection_service!=None :
            try:
                self.detection_service.load_model(model=model)
                return jsonify(result='DONE LOADING SUCCESS')
            except:
                return jsonify(error='ERROR model throw exception')

        return jsonify(result='ERROR model is null')