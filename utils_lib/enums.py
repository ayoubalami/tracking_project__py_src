
from enum import Enum


class StreamSourceEnum(Enum):
    FILE = 1
    WEBCAM = 2
    RASPBERRY_CAM=4

class ClientStreamTypeEnum(Enum):
    CNN_DETECTOR = 1
    BACKGROUND_SUBTRACTION = 2
    TRACKING_STREAM=3
    HYBRID_TRACKING_STREAM=4
 
class SurveillanceRegionEnum(Enum):
    DETECTION_REGION = 1
    TRACKING_REGION = 2
    PRETRACKING_REGION = 3

