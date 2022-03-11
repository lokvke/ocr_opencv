from config.cfg import *
from net.text_detection import TextDetection
from net.text_recognition import TextRecognition


# text recognition
tr = TextRecognition(TEXT_RECOGNITION_MODEL, ALPHABET)

# text detection
td = TextDetection(TEXT_DETECTION_MODEL)


def text_recognition(image):
    outputs = tr(image)
    return outputs


def text_detection(image):
    outputs = td(image)
    return outputs





