import os
import cv2
import matplotlib.image as mpimg
from datetime import datetime
from LaneFindingAlgorithm import LaneFindingAlgorithm


def processing(inputpath, outputpath, debug):
    """Reads every image in the director and performs back projection to find ROI
       Also calls LaneFindingAlgorithm to store the output images in the output location"""
    test_files = os.listdir(inputpath)
    # Sort the file names in alphanumeric order
    test_files.sort()
    total_count = float(len(test_files))
    start = datetime.now()
    leftHistory = []
    rightHistory = []

    for filename in test_files:

        print "processing: ", filename
        # read using matplolib to send as a parameter to the lane finding algorithm
        input_image = mpimg.imread(inputpath + filename)


    #number of rows divided by number of columns (1080/1920)
        # dim = (400, 255)
        # resized = cv2.resize(input_image, dim, interpolation=cv2.INTER_AREA)
        # print resized.shape




        # ****************************************THIS IS HISTOGRAM BACK PROJECION***********************************
        # target = cv2.imread(inputpath + filename)
        #
        # # [y1:y2, x1:x2]  - x axis is the horizontal length and y axis is the vertical length of the picture
        # # select the road texture - the one which we want to back project
        # roi = target[800:1080, 850:1200]
        # # if debug:
        # #     cv2.imshow('roi', roi)
        # #     cv2.waitKey(0)
        # #     cv2.destroy
        # hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # hsvt = cv2AllWindows()
        #.cvtColor(target, cv2.COLOR_BGR2HSV)
        #
        # # calculating object histogram
        # roihist = cv2.calcHist([hsv], [0, 1], None, [256, 256], [0, 256, 0, 256])
        #
        # # normalize histogram and apply back projection
        # cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
        # dst = cv2.calcBackProject([hsvt], [0, 1], roihist, [0, 256, 0, 256], 1)
        #
        # # Now convolute with circular disc
        # disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # cv2.filter2D(dst, -1, disc, dst)
        #
        # # threshold and binary
        # ret, thresh = cv2.threshold(dst, 0, 255, 0)
        # thresh = cv2.merge((thresh, thresh, thresh))
        # # if debug:
        # #     cv2.imshow('thresh', thresh)
        # #     cv2.waitKey(0)
        # #     cv2.destroyAllWindows()
        #
        # res = cv2.bitwise_and(target, thresh)
        # # if debug:
        #     # cv2.imshow('res', res)
        #     # cv2.waitKey(0)
        #     # cv2.destroyAllWindows()
        #
        # # print res.shape[0]    -- this is 1080 for all images
        # rowROI = 0
        # for i in range(0, res.shape[0], 3):
        #     count = 0
        #     for j in range(res.shape[1]):
        #         # count the number of pixels which are not black
        #         if res[i, j][0] != 0:
        #             count += 1
        #
        #     if count > 500:
        #         rowROI = i
        #         # print count
        #         break
        #
        # # print rowROI
        # if rowROI < 540:
        #     rowROI = 540

        # ******************************BACK PROJECTION ENDS HERE**************************************


        rowROI = input_image.shape[0]/2
        output, leftHistory, rightHistory = LaneFindingAlgorithm.findLanes(input_image, rowROI, leftHistory,
                                                                           rightHistory, debug)
        mpimg.imsave(outputpath + filename, output, format='jpg')
        print "*****************************************************"
        print "processing of file " + filename + " is complete."


    end = datetime.now()
    total_time = (end - start).total_seconds()
    frame_rate = total_count / total_time
    print("\nTotal time for " + str(int(total_count)) + " images is " + str(total_time) + " seconds")
    print("\nFrame rate: " + str(frame_rate) + " images per second")


# Execution starts here
if __name__ == '__main__':
    input_directory = "/home/ashwani/Desktop/testingInput/"
    output_directory = "/home/ashwani/Desktop/testingOutputAshwaniCode/"
    debug = False
    processing(input_directory, output_directory, debug)
