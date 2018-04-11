import os
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from datetime import datetime
from LaneFindingAlgorithm import LaneFindingAlgorithm
from picamera import PiCamera
from time import sleep
from picamera.array import PiRGBArray


def run_for_camera():
    leftHistory = []
    rightHistory = []
    
    camera = PiCamera()
    camera.resolution = (256,144)
    camera.framerate = 5
    camera.iso = 800
    camera.saturation = 35
    camera.sharpness = 10
    camera.video_stabilization = True
    rawCapture = PiRGBArray(camera, size = (256,144))
    cv2.namedWindow("Lane")
    cv2.setWindowProperty("Lane", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # cv2.resizeWindow("Lane", 720, 400)
    sleep(0.2)

    for frame in camera.capture_continuous(rawCapture, format = "bgr", use_video_port = True):
        input_Image = frame.array
        rowROI = input_Image.shape[0]/2
        output, leftHistory, rightHistory = LaneFindingAlgorithm.findLanes(input_Image, rowROI, leftHistory,
                                                                           rightHistory, False)

        
        cv2.imshow("Lane", output)
        rawCapture.truncate(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 
        del frame
        del output
        


# Execution starts here
if __name__ == '__main__':
    run_for_camera()
