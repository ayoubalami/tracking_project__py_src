# from turtle import shape
import threading
from flask import jsonify,stream_with_context,Flask,render_template,Response
from classes.buffer import Buffer
from classes.tensorflow_detection_service import TensorflowDetectionService
from classes.stream_reader import StreamReader
from classes.detection_service import IDetectionService


class AppService:
    
    stream_reader :StreamReader = None
    buffer :Buffer= None
    detection_service :IDetectionService= None
    video_src = "videos/highway2.mp4"
    buffering_thread=None

    def __init__(self):
        print("AppService Starting ...")

        self.detection_service=TensorflowDetectionService()
        
        if self.detection_service!=None :
            print( " detection_module loaded succesufuly")
            print( "Service name : ",self.detection_service.service_name())
            print( "Model name : ", self.detection_service.model_name())
        else :
            print( " No detection_module To load")

        print("AppService Started.")

        
    def read_stream(self):
        yield from self.stream_reader.readStream()
        # pass
 
    def index(self):
        return render_template('index.html')
 
    def video_stream(self):

        # youtube_url = "https://www.youtube.com/watch?v=QuUxHIVUoaY"
        # buffer=Buffer(youtube_url=youtube_url)
       
        self.buffer=Buffer(video_src=self.video_src)
        self.stream_reader=StreamReader(self.buffer,self.detection_service)

        self.buffering_thread= threading.Thread(target=self.buffer.downloadBuffer)
        self.buffering_thread.start()
        return Response(self.read_stream(),mimetype='multipart/x-mixed-replace; boundary=frame')

    def timer(self):
        if self.stream_reader != None :
            return jsonify(result=self.stream_reader.current_time)
        else :
            return jsonify(result=1)

    def videoDuration(self):
        return jsonify(result=self.buffer.video_duration)

    def clean_memory(self):
        if self.buffer :
            self.buffer.stop_buffring_event.set()
            return jsonify(result='DONE')
        return jsonify(result='clean_memory ERROR')

    def stopStream(self):
        if self.buffer and not self.stream_reader.stop_reading_from_user_action :
            self.stream_reader.stop_reading_from_user_action=True
            return jsonify(result='stream stoped')
        return jsonify(result='error server in stream stoped')

    def startStream(self):
        if self.buffer and self.stream_reader.stop_reading_from_user_action :
            self.stream_reader.stop_reading_from_user_action=False
            return jsonify(result='stream started')
        return jsonify(result='error server in stream started')

    
