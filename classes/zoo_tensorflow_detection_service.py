
from http import server
import cv2,time,os,numpy as np
from classes.detection_service import IDetectionService
 
from tensorflow.python.keras.utils.data_utils import get_file 
  # Suppress TensorFlow logging
import tensorflow as tf

from object_detection.utils import config_util
from object_detection.builders import model_builder

# from object_detection.utils import label_map_util
# from object_detection.utils import config_util
# from object_detection.utils import visualization_utils as viz_utils
# from object_detection.builders import model_builder
import tarfile
import urllib.request

class ZooTensorflowDetectionService(IDetectionService):

    np.random.seed(123)
    # threshold = 0.5
     
    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
        self.DATA_DIR = os.path.join(os.getcwd(), 'storages')
        self.MODELS_DIR = os.path.join(self.DATA_DIR, 'tensorflow_models')
        for dir in [self.DATA_DIR, self.MODELS_DIR]:
            if not os.path.exists(dir):
                os.mkdir(dir)
        self.modelName="object detection API"
        self.load_model()
        # pass
        
    def service_name(self):
        return "Tensorflow detection service V 2.0"

    def model_name(self):
        return self.modelName
        
    def load_model(self):
        self.load_or_download_model_tensorflow()

    def load_or_download_model_tensorflow(self):
        import tarfile
        import urllib.request

    # Download and extract model
        self.MODEL_DATE = '20200711'
        self.MODEL_NAME = 'ssd_mobilenet_v2_320x320_coco17_tpu-8'
        self.MODEL_TAR_FILENAME = self.MODEL_NAME + '.tar.gz'
        self.MODELS_DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/tf2/'
        self.MODEL_DOWNLOAD_LINK = self.MODELS_DOWNLOAD_BASE + self.MODEL_DATE + '/' + self.MODEL_TAR_FILENAME
        self.PATH_TO_MODEL_TAR = os.path.join(self.MODELS_DIR, self.MODEL_TAR_FILENAME)
        self.PATH_TO_CKPT = os.path.join(self.MODELS_DIR, os.path.join(self.MODEL_NAME, 'checkpoint/'))
        self.PATH_TO_CFG = os.path.join(self.MODELS_DIR, os.path.join(self.MODEL_NAME, 'pipeline.config'))
        if not os.path.exists(self.PATH_TO_CKPT):
            print('Downloading model. This may take a while... ', end='')
            urllib.request.urlretrieve(self.MODEL_DOWNLOAD_LINK, self.PATH_TO_MODEL_TAR)
            tar_file = tarfile.open(self.PATH_TO_MODEL_TAR)
            tar_file.extractall(self.MODELS_DIR)
            tar_file.close()
            os.remove(self.PATH_TO_MODEL_TAR)
            print('Done')

        # Download labels file
        self.LABEL_FILENAME = 'mscoco_label_map.pbtxt'
        self.LABELS_DOWNLOAD_BASE = \
            'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
        self.PATH_TO_LABELS = os.path.join(self.MODELS_DIR, os.path.join(self.MODEL_NAME, self.LABEL_FILENAME))
        if not os.path.exists(self.PATH_TO_LABELS):
            print('Downloading label file... ', end='')
            urllib.request.urlretrieve(self.LABELS_DOWNLOAD_BASE + self.LABEL_FILENAME, self.PATH_TO_LABELS)
            print('Done')
            pass
    



 

        # tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

        # # Enable GPU dynamic memory allocation
        # gpus = tf.config.experimental.list_physical_devices('GPU')
        # for gpu in gpus:
        #     tf.config.experimental.set_memory_growth(gpu, True)

        # # Load pipeline config and build a detection model
        # configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
        # model_config = configs['model']
        # detection_model = model_builder.build(model_config=model_config, is_training=False)

        # # Restore checkpoint
        # ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        # ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

        # @tf.function
        # def detect_fn(image):
        #     """Detect objects in image."""

        #     image, shapes = detection_model.preprocess(image)
        #     prediction_dict = detection_model.predict(image, shapes)
        #     detections = detection_model.postprocess(prediction_dict, shapes)

        #     return detections, prediction_dict, tf.reshape(shapes, [-1])


# sed -i 's/slim = tf.contrib.slim/import tf_slim as slim' mask_head.py


    def detect_objects(self, frame,threshold= 0.5):
        return frame



    # sed -i 's/slim = tf.contrib.slim/import tf_slim as slim/g' mask_head.py
    
    # find /usr/local/lib/python3.9/dist-packages/object_detection -type f -exec sed -i 's/slim = tf.contrib.slim/import tf_slim as slim/g' {} +
