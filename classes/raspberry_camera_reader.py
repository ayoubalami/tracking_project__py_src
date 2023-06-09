

import json
import time
import cv2
from utils_lib.enums import ProcessingTaskEnum
import base64
from servo_motor import ServoMotor

# from utils_lib.utils_functions import encodeStreamingFrame,applyDetection

from classes.background_subtractor_service import BackgroundSubtractorService
from classes.tracking_service.tracking_service import TrackingService
from classes.detection_services.detection_service import IDetectionService

class RaspberryCameraReader :
    def __init__(self,detection_service:IDetectionService,background_subtractor_service:BackgroundSubtractorService,tracking_service:TrackingService): 
        from picamera2 import Picamera2, Preview
        from picamera2.encoders import H264Encoder
        self.start_reading_action=False
        self.processing_task:ProcessingTaskEnum= ProcessingTaskEnum.CNN_DETECTOR
        self.jpeg_compression_ratio=75
        self.detection_service=detection_service
        self.background_subtractor_service=background_subtractor_service
        self.tracking_service=tracking_service
        self.tracking_service.raspberry_camera=self
        # self.threshold=0.5
        # self.nms_threshold=0.5
        self.zoom=1
        self.y_servo_motor=ServoMotor(servo_pin=18)
        self.x_servo_motor=ServoMotor(servo_pin=23)  
        self.tracked_object=None
        self.frame_size=None
        self.terminate_tracking=False

        try:
            self.x_servo_motor.goToAngleWithSpeed(angle=int(0),speed=0.001 )
        except:
            self.x_servo_motor.goToAngle(angle=0 )  
        try:
            self.y_servo_motor.goToAngleWithSpeed(angle=int(80),speed=0.001 )
        except:
            self.y_servo_motor.goToAngle(angle=80 )

        try:
            self.picam2 = Picamera2()
        except:
            print("camera not detected program exited....")
            exit()
        self.picam2.configure(self.picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (1024, 768)}))
        time.sleep(.1)

    def read_camera_stream(self):
        print("Start READING FROM raspberry camera.||.....")
        try:
            self.picam2.start()
        except:
            self.picam2.stop()
            time.sleep(.2)
            self.picam2.start()

        while(True):
            if self.start_reading_action==False:
                time.sleep(.1)
                continue
            image = self.picam2.capture_array()
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            self.frame_size=image.shape

            if self.zoom>1:
                height,width=image.shape[:2]
                x=int((width*.5)- (width/self.zoom)*.5)
                y=int((height*.5)- (height/self.zoom)*.5)
                w=int(width/self.zoom)
                h=int(height/self.zoom)
                image = image[y:y+h, x:x+w]
                self.frame_size=image.shape


            # detection_frame,inference_time=self.applyDetection(image,self.detection_service,threshold=self.threshold,nms_threshold=self.nms_threshold)

            if not self.detection_service :
                time.sleep(.02)
            yield from self.ProcessAndYieldFrame(image)
      
        # self.picam2.stop()
        # print("Stop READING FROM raspberry camera.||.....")
    
    def ProcessAndYieldFrame(self,frame):
        result={}
        copy_frame=frame.copy()

        # copy_frame=copy_frame[:,:,1:4]
        # copy_frame=copy_frame[:,:,:3]

        if self.processing_task== ProcessingTaskEnum.CNN_DETECTOR:
            detection_frame,inference_time=self.applyDetection(copy_frame)
            result['detectorStream']=self.encodeStreamingFrame(frame=detection_frame,resize_ratio=1,jpeg_quality=self.jpeg_compression_ratio)

        elif self.processing_task== ProcessingTaskEnum.BACKGROUND_SUBTRACTION:
            merged_foreground_detection_frame,resized_foreground_detection_frame,raw_mask_frame,inference_time=self.background_subtractor_service.apply(copy_frame)
            result['backgroundSubStream_1']=self.encodeStreamingFrame(frame=raw_mask_frame,resize_ratio=1,jpeg_quality=self.jpeg_compression_ratio)
            # result['backgroundSubStream_2']=self.encodeStreamingFrame(frame=resized_foreground_detection_frame,resize_ratio=1,jpeg_quality=self.jpeg_compression_ratio)
            result['backgroundSubStream_3']=self.encodeStreamingFrame(frame=merged_foreground_detection_frame,resize_ratio=1,jpeg_quality=self.jpeg_compression_ratio)

        elif self.processing_task== ProcessingTaskEnum.TRACKING_STREAM:
            tracking_frame=self.tracking_service.apply(copy_frame)
            result['trackingStream_1']=self.encodeStreamingFrame(frame=tracking_frame,resize_ratio=1,jpeg_quality=self.jpeg_compression_ratio)

        # elif self.current_selected_stream== ClientStreamTypeEnum.HYBRID_TRACKING_STREAM:
        #     tracking_frame=self.tracking_service.apply(copy_frame,threshold= self.threshold ,nms_threshold=self.nms_threshold)
        #     result['hybridTrackingStream_1']=self.encodeStreamingFrame(frame=tracking_frame,resize_ratio=1,jpeg_quality=self.jpeg_compression_ratio)
         
        yield 'event: message\ndata: ' + json.dumps(result) + '\n\n'

    def encodeStreamingFrame(self,frame,resize_ratio=1,jpeg_quality=100):
        if resize_ratio!=1:
            img_width, img_height = frame.shape[1], frame.shape[0]
            frame=cv2.resize(frame, (int(img_width*resize_ratio) ,int(img_height*resize_ratio) ))
        ret,buffer=cv2.imencode('.jpg',frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        img_bytes=buffer.tobytes()
        return  base64.b64encode(img_bytes).decode()

    def applyDetection(self,origin_frame):   
        # ret,origin_frame=cv2.imencode('.jpg',origin_frame, [cv2.IMWRITE_JPEG_QUALITY, 30])
        # resize_ratio=.5
        # origin_frame=cv2.resize(origin_frame, (int(self.buffer.width*resize_ratio) ,int(self.buffer.height*resize_ratio) ))
        if self.detection_service !=None  and self.detection_service.get_selected_model() !=None:
            detection_frame ,inference_time = self.detection_service.detect_objects(origin_frame)
            return detection_frame,inference_time
        return origin_frame,-1

    def rotateServoMotor(self,axis,angle,speed=0.005):
        if axis=='y':    
            self.y_servo_motor.goToAngleWithSpeed(angle=angle,speed=speed) 
        if axis=='x':    
            self.x_servo_motor.goToAngleWithSpeed(angle=angle,speed=speed) 


    def moveServoMotorToCoordinates(self,track,speed=0.005):
        (heigth,width)=self.frame_size[:2]
        centerX=int(width/2)
        centerY=int(heigth/2)
        epsilon=10
        step=1
        print("TRACK OBJECT ID "+str(track.track_id))
        TTL=1000
        while(True):
            if self.terminate_tracking:
                break
            bbox = track.to_tlbr()
            (dest_x,dest_y) = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
            # print(" TRACK UPDATED : " + str(track.time_since_update)+ " - " +str(track.track_id)+" - "+str(track.to_tlbr()))   
            
            # TrackState.Deleted == 3
            if track.state == 3:
                print("TRACK IS DELETED")
                break    

            distance=abs(centerY-dest_y)
            # print("distance Y : " , str( distance))
            if distance>epsilon: 
                step=self.calculateStep(distance)                    
                if centerY>dest_y :
                    self.y_servo_motor.moveToUp(step=step)
                else :
                    self.y_servo_motor.moveToDown(step=step) 
            else:
                pass                 

            distance=abs(centerX-dest_x)
            # print("distance X : " , str( distance))
            if distance>epsilon: 
                step=self.calculateStep(distance)
                if centerX>dest_x :
                    self.x_servo_motor.moveToRight(step=step)
                else :
                    self.x_servo_motor.moveToLeft(step=step) 
            else:
                # print(" TARGET DONE :")
                pass
                # break

            time.sleep(.1)
        self.terminate_tracking=False



    def calculateStep(self,distance):
        step=1
        if distance<15:
            step=.5
        elif distance<20:
            step=.75
        elif distance<26:
            step=1
        elif distance<30:
            step=1.25
        elif distance<40:
            step=2
        elif distance<50:
            step=2.25
        elif distance<80:
            step=3.5
        elif distance<140:
            step=4
        else:
            step=5
        return step
