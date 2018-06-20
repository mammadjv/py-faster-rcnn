from get_plates_location import PositionDetector
import cv2
import time

pos = PositionDetector()
im = cv2.imread("/home/mj/workspace/py-faster-rcnn/1.png")

start = time.time()
pos.detect(im)
print time.time()- start
