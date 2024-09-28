
from enum import Enum


class TrackingMethodEnum(Enum):
    DEEP_SORT=0
    SORT=1
    ANCHOR_BASED=2
    

class StreamSourceEnum(Enum):
    FILE = 1
    WEBCAM = 2
    RASPBERRY_CAM=4

class ClientStreamTypeEnum(Enum):
    STREAMING = 1
    CNN_DETECTOR = 2
    BACKGROUND_SUBTRACTION = 3
    TRACKING_STREAM=4
    HYBRID_TRACKING_STREAM=5

class ProcessingTaskEnum(Enum):
    RAW_STREAM = 0
    CNN_DETECTOR = 1
    BACKGROUND_SUBTRACTION = 2
    TRACKING_STREAM=3
    HYBRID_TRACKING_STREAM=4
 
class DetectorForTrackEnum(Enum):
    CNN_DETECTOR = 1
    BACKGROUND_SUBTRACTION = 2

class SurveillanceRegionEnum(Enum):
    DETECTION_REGION = 1
    TRACKING_REGION = 2
    # PRETRACKING_REGION = 3


class ObjectStatusEnum(Enum):
    DETECTED = 1
    TRACKED = 2
    MISSED = 3
    STOPPED = 4
    EXITED=5
