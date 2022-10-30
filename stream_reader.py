
import cv2,time,os, numpy as np
from buffer import Buffer
import pyshine as ps
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import get_file 


class StreamReader: 

    np.random.seed(123)
    # threshold = 0.5
    perf = []
    classAllowed=[]
    colorList=[]

    def __init__(self, buffer:Buffer):
        # self.target_delay=buffer.frame_duration
        
        self.buffer=buffer
        self.current_time=0
        self.current_frame_index=0
        self.stop_reading_for_loading=False
        self.stop_reading_from_user_action=False
        self.end_of_file=False
        self.buffer.stream_reader=self
        # https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
        self.modelURL= "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
        # self.modelURL= "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz"
        # self.modelURL= "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz"
        # self.modelURL="http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz"
       #### self.modelURL="http://download.tensorflow.org/models/object_detection/tf2/20210210/centernet_mobilenetv2fpn_512x512_coco17_od.tar.gz"
        # self.modelURL="http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v2_512x512_coco17_tpu-8.tar.gz"
        # self.modelURL="http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_640x640_coco17_tpu-8.tar.gz"

        self.init_download_model_tensorflow() 

    def clean_streamer(self):
        print("STREAM_READER : begin clean_streamer")
        del(self.buffer.buffer_frames)
        self.buffer.cap.release()
        del(self.buffer.cap)
        print("STREAM_READER : end clean_streamer")

    

    def readStream(self):
        delay=self.buffer.frame_duration
        # delay for buffer to load some frame
        time.sleep(0.1)
        current_sec=0
        decalage=0
        while True :

            if self.stop_reading_from_user_action:
                time.sleep(.2)
                continue

            if (self.buffer.stop_buffring_event.is_set()):
                self.clean_streamer()
                print("STREAM_READER : end_of_thread")
                break; 

            t1= time.time()
            if self.current_frame_index>=len(self.buffer.buffer_frames) or len(self.buffer.buffer_frames)==0 :
                continue

            frame,current_batch=self.getCurrentFrame()
               
            frame =  ps.putBText(frame,str(current_sec),text_offset_x=20,text_offset_y=20,vspace=10,hspace=10, font_scale=2.0,background_RGB=(210,220,222),text_RGB=(255,250,250))
            ret,buffer=cv2.imencode('.jpg',frame)
            img_bytes=buffer.tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')
            self.current_time=self.current_time+self.buffer.frame_duration
            new_sec=self.current_time
            if new_sec>=current_sec+1:
                current_sec=current_sec+1
                new_sec=current_sec
            
            # print("***************")
            # print(" TOTAL FRAMES IN MEMORY : "+str(len(self.buffer.buffer_frames)))
            # print("current_frame_index : "+str(self.current_frame_index))
            # print("current_batch : "+ str(current_batch))
            # print("last_batch : "+ str(self.buffer.last_batch))
            # print("last_frame_of_last_batch : "+ str(self.buffer.last_frame_of_last_batch))
            # print("==============")

            self.current_frame_index=self.current_frame_index+1
            # DELETE PREVIOUS BATCH FRAMES ALREADY PRINTED FROM MEMORY
            if (self.current_frame_index==self.buffer.batch_size):
                self.buffer.delete_last_batch(current_batch)
                self.current_frame_index=0
                self.buffer.download_new_batch=True
                # print("<<<<<< decalage >>>>" + str(decalage))
            
            # END READING IN CASE OF REACHING THE LAST BATCH
            if ( self.buffer.last_frame_of_last_batch==self.current_frame_index  and current_batch==self.buffer.last_batch):
                self.clean_streamer()
                print("STREAM_READER : end_of_file")
                break;   

            # CALCULATE TIME OF INSTRUCTIONS AND ADD DELAY TO REACH THE FRAME DURATION 
            calculeTime=(time.time()-t1)
            delay=self.buffer.frame_duration-calculeTime + decalage
            if delay>0:
                decalage=0
                time.sleep(delay)
            else:
                if delay<0 :
                    decalage=delay


    def getCurrentFrame(self):
        
        frame,current_batch= self.buffer.buffer_frames[self.current_frame_index]  

        frame = self.imageDetectionTensorflow(frame, 0.4)
   
        # detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # faces=detector.detectMultiScale(frame,1.1,7)
        #     #Draw the rectangle around each face
        # for (x, y, w, h) in faces:
        #     cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
         
        return frame,current_batch


    def readClasses(self, classesFilePath): 
        global classesList
        global colorList
        global classAllowed
        classAllowed=[0,1,2,3,5,6,7]
        
        with open(classesFilePath, 'r') as f:
            classesList = f.read().splitlines()
            
        #   delete all class except person and vehiccule 
        classesList=classesList[0:8]
        classesList.pop(4)
        print(classesList)
    
        # Colors list 
        colorList =  [[23.82390253, 213.55385765, 104.61775798],
            [168.73771775, 240.51614241,  62.50830085],
            [  3.35575698,   6.15784347, 240.89335156],
            [235.76073062, 119.16921962,  95.65283276],
            [138.42940829, 219.02379358, 166.29923782],
            [ 59.40987365, 197.51795215,  34.32644182],
            [ 42.21779254, 156.23398212,  60.88976857]]
        classesList.insert(0,-1)
        colorList.insert(0,-1)
        

    def init_download_model_tensorflow(self):

        fileName = os.path.basename(self.modelURL) 
        global modelName
        global cacheDir
        global model
        global is_tensorflow_model

        classFile ="models/coco.names" 
    
        is_tensorflow_model=True
        modelName = fileName[:fileName.index('.')]
        cacheDir = os.path.join("models","tensorflow_models", modelName)

        os.makedirs(cacheDir, exist_ok=True)
        get_file(fname=fileName,origin=self.modelURL, cache_dir=cacheDir, cache_subdir="checkpoints",  extract=True)
        tf.keras.backend.clear_session()
        model = tf.saved_model.load(os.path.join(cacheDir, "checkpoints", modelName, "saved_model"))
        print("Model " + modelName + " loaded successfully...")
        self.readClasses(classFile)


    def imageDetectionTensorflow(self, frame,threshold= 0.5,setSoftNMS=False):
        global model
        global classesList
        global colorList
        global detections

        inputTensor = cv2.cvtColor( frame.copy(), cv2.COLOR_BGR2RGB ) 
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8) 
        inputTensor = inputTensor[tf.newaxis,...]
        detections = model(inputTensor)
        bboxs = detections['detection_boxes'][0].numpy()
        classIndexes = detections['detection_classes'][0].numpy().astype(np.int32) 
        classScores = detections['detection_scores'][0].numpy()
        imH, imW, imC = frame.shape

        if setSoftNMS :
            bboxIdx = tf.image.non_max_suppression_with_scores(bboxs, classScores, max_output_size=150, 
            iou_threshold=threshold, soft_nms_sigma=threshold)
        else :
            bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=150, 
            iou_threshold=threshold, score_threshold=threshold)

        if len(bboxIdx) != 0: 
            for i in bboxIdx:
                bbox = tuple(bboxs[i].tolist())
                classConfidence = round(100*classScores[i])
                classIndex = classIndexes[i]

                if classIndex not in classAllowed:
                    continue

                classLabelText=classesList[classAllowed.index(classIndex)]
                classColor = colorList[classAllowed.index(classIndex)]

                # classColor = colorList[classIndex]
                displayText = '{}: {}'.format(classLabelText, classConfidence) 
                ymin, xmin, ymax, xmax = bbox
                xmin, xmax, ymin, ymax = (xmin * imW, xmax * imW, ymin * imH, ymax * imH) 
                xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=classColor, thickness=2) 
                cv2.putText(frame, displayText, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

        return frame


    # def imageDetectionTensorflow( image,threshold= 0.5,videoFrame=False,imagePath=None,setSoftNMS=False):
    #     global model
    #     global classesList
    #     global colorList
    #     global detections

    #     inputTensor = cv2.cvtColor( image.copy(), cv2.COLOR_BGR2RGB ) 
    #     inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8) 
    #     inputTensor = inputTensor[tf.newaxis,...]

    #     s = time.time()
    #     detections = model(inputTensor)
    #     curr_time = (time.time()-s )*1000
    #     if videoFrame==False :
    #         print(modelName )
    #         print("execution time : ",str(curr_time))
        
    #     bboxs = detections['detection_boxes'][0].numpy()
    #     classIndexes = detections['detection_classes'][0].numpy().astype(np.int32) 
    #     classScores = detections['detection_scores'][0].numpy()
    #     imH, imW, imC = image.shape

    #     if setSoftNMS :
    #         bboxIdx = tf.image.non_max_suppression_with_scores(bboxs, classScores, max_output_size=150, 
    #         iou_threshold=threshold, soft_nms_sigma=threshold)
    #     else :
    #         bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=150, 
    #         iou_threshold=threshold, score_threshold=threshold)

    #     if imagePath != None:
    #         print("add pref")
    #         imagePerf={"framework":"Tensorflow","imagePath":imagePath,"exec_time":curr_time,"modelName":modelName,"object_detected":len(bboxIdx)}
    #         perf.append(imagePerf)

    #     if len(bboxIdx) != 0: 
    #         for i in bboxIdx:
    #             bbox = tuple(bboxs[i].tolist())
    #             classConfidence = round(100*classScores[i])
    #             classIndex = classIndexes[i]

    #             if (classIndex in classAllowed)==False:
    #                 continue

    #             classLabelText=classesList[classAllowed.index(classIndex)]
    #             classColor = colorList[classAllowed.index(classIndex)]

    #             # classColor = colorList[classIndex]
    #             displayText = '{}: {}'.format(classLabelText, classConfidence) 
    #             ymin, xmin, ymax, xmax = bbox
    #             xmin, xmax, ymin, ymax = (xmin * imW, xmax * imW, ymin * imH, ymax * imH) 
    #             xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

    #             cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=classColor, thickness=2) 
    #             cv2.putText(image, displayText, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)
    #             ###############################################
    #     return image 
           
