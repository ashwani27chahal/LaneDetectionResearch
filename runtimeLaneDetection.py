import os
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from datetime import datetime
from LaneFindingAlgorithm import LaneFindingAlgorithm
# from picamera import PiCamera
import time
from time import sleep
# from picamera.array import PiRGBArray


def run_for_camera():
    leftHistory = []
    rightHistory = []
    
    camera = PiCamera()
    camera.resolution = (400,225)
    camera.framerate = 10
    camera.iso = 800
    camera.saturation = 35
    camera.sharpness = 10
    camera.video_stabilization = True
    rawCapture = PiRGBArray(camera, size = (400,225))
    cv2.namedWindow("Lane")
    cv2.setWindowProperty("Lane", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow("Lane", 1020, 980)
    sleep(0.1)

    for frame in camera.capture_continuous(rawCapture, format = "bgr", use_video_port = True):
        input_Image = frame.array
        start = time.time()
        rowROI = input_Image.shape[0]/2
        output, leftHistory, rightHistory = LaneFindingAlgorithm.findLanes(input_Image, rowROI, leftHistory,
                                                                           rightHistory, False)

        
        cv2.imshow("Lane", output)
        end = time.time()
        print("Frames per second:", 1 / (end - start))      
        rawCapture.truncate(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 
        del frame
        del output

def videoReading(videoFileName):
    leftHistory = []
    rightHistory = []
    
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(videoFileName)
# Check if camera opened successfully
#  if (cap.isOpened() == False):
#    print("Error opening video stream or file")

    # Read until video is completed
    frameCounter = 0
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            input_Image = frame
            start = time.time()
            rowROI = input_Image.shape[0] / 2
            output, leftHistory, rightHistory = LaneFindingAlgorithm.findLanes(input_Image, rowROI, leftHistory,
                                                                               rightHistory, False)
            cv2.imshow("Lane", output)
            end = time.time()
            print("Frames per second:", 1 / (end - start))

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

# Execution starts here
if __name__ == '__main__':
    videoReading("trafficClip1_small.mp4")
    # run_for_camera()
