
from enum import Enum


class StreamSourceEnum(Enum):
    FILE = 1
    WEBCAM = 2
    YOUTUBE=3

class ClientStreamTypeEnum(Enum):
    CNN_DETECTOR = 1
    BACKGROUND_SUBTRACTION = 2
 