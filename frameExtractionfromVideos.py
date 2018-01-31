import numpy as np
import sys
import cv2

fps = 10
frame_count = 1
total_frame_count = 1800.0 * fps

cap = cv2.VideoCapture("/home/ashwani/Desktop/drive Jan 26 2018 - snow data videos/video2.mp4")
cap.set(cv2.CAP_PROP_FPS, fps);

while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imwrite("/home/ashwani/Desktop/frameCollection/output"+str(frame_count)+".jpg", frame)
    frame_count += 1
    sys.stdout.write('\r')
    progress = (frame_count / total_frame_count) * 100;
    sys.stdout.write("Current frame: %d Progress: %.3f%%" % (frame_count, progress))
    sys.stdout.flush()
    del frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


#
# -------
# Converting .h264 to mp4:   MP4Box -add filename.h264 filename.mp4
#
# If MP4Box is not installed then install it first -
# Installation
#
# sudo apt-get update
# sudo apt-get install gpac
# y
# Usage