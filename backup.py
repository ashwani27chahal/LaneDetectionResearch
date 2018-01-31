import matplotlib.pyplot as plt
import numpy as np
import cv2
import csv

"""This class masks image based on ROI in the parameter, computes edges, and draw hough lines on the original image"""


class LaneFindingAlgorithm:
    def __init__(self):
        pass





    @staticmethod
    def grayscale(img):
        """Applies Grayscaling"""
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)






    @staticmethod
    def abs_sobel_thresh(img, thresh_min, thresh_max):
        """Computes edges in x-direction gradient change"""
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))

        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        return binary_output





    @staticmethod
    def gaussian_blur(img, kernel_size):
        """Applies Gaussian Blurring"""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)






    @staticmethod
    def region_of_interest(img, vertices):
        """Applies an image mask.Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black"""
        # defining a blank mask to start with
        mask = np.zeros_like(img)
        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image








    @staticmethod
    def draw_lines(img, lines, leftHistory, rightHistory, minAngleLeftLanes, minAngleRightLanes,
                                                                    maxAngleLeftLanes, maxAngleRightLanes, flag):

        """Draws lined found by hough on the image and filters out the lines based on their slope and y-intercept
        Also, determines the left lane history and right lane history"""

        if lines is None:
            if flag:
                print "using history here"
            for lane in leftHistory:
                if flag:
                    print "left lane lines in history"
                    print lane
                for x1, y1, x2, y2 in lane:
                    cv2.line(img, (x1, y1), (x2, y2), [255, 255, 0], 11)

            for lane in rightHistory:
                if flag:
                    print "right lane lines in history"
                    print lane
                for x1, y1, x2, y2 in lane:
                    cv2.line(img, (x1, y1), (x2, y2), [255, 255, 0], 11)

            return leftHistory, rightHistory
        if flag:
            print "these are all the lines created by hough transform"
            print lines

        leftLaneLines = []
        rightLaneLines = []

        leftLineIntercept = []
        rightLineIntercept = []
        for line in lines:

            for x1, y1, x2, y2 in line:
                if (x2 - x1) == 0:
                    continue
                m = (((img.shape[0] - y2) - (img.shape[0] - y1)) * 1.0) / (x2 - x1)
                c = (img.shape[0] - y2) - (m * x2 * 1.0)

                if flag:
                    print "slope of this line is:", m
                angle = ((np.arctan(m)) / np.pi) * 180

                if flag:
                    print "angle of line in degrees is:", angle

                if minAngleLeftLanes < angle < maxAngleLeftLanes:
                    leftLaneLines.append(line)
                    leftLineIntercept.append(c)

                if minAngleRightLanes < angle < maxAngleRightLanes:
                    rightLaneLines.append(line)
                    rightLineIntercept.append(c)

        if flag:
            print "Left lane lines: ", leftLaneLines
            print "Left history: ", leftHistory

        leftFlag = True
        if leftLaneLines == []:
            leftFlag = False

        if leftFlag:
            outputLeftLanes = []
            meanLeft = np.mean(leftLineIntercept)

            for leftLine in leftLaneLines:
                for x1, y1, x2, y2 in leftLine:
                    if (x2 - x1) == 0:
                        continue
                    m = (((1080 - y2) - (1080 - y1)) * 1.0) / (x2 - x1)
                    cLeft = (1080 - y2) - (m * x2 * 1.0)
                    if (meanLeft + 150) >= cLeft >= (meanLeft - 150):
                        outputLeftLanes.append(leftLine)

            if flag:
                print "output left lanes: ", outputLeftLanes

            if outputLeftLanes == []:
                leftFlag = False

        if leftFlag:
            leftHistory = np.copy(outputLeftLanes)
            for lane in outputLeftLanes:
                for x1, y1, x2, y2 in lane:
                    cv2.line(img, (x1, y1), (x2, y2), [0, 255, 0], 10)

        if not leftFlag:

            if flag:
                print "using history here"

            for lane in leftHistory:

                for x1, y1, x2, y2 in lane:
                    cv2.line(img, (x1, y1), (x2, y2), [255, 255, 0], 11)

        if flag:
            print "Right lane lines: ", rightLaneLines
            print "Right history: ", rightHistory
        rightFlag = True
        if rightLaneLines == []:
            rightFlag = False

        if rightFlag:

            outputRightLanes = []
            meanRight = np.mean(rightLineIntercept)


            for rightLine in rightLaneLines:
                for x1, y1, x2, y2 in rightLine:
                    if (x2 - x1) == 0:
                        continue
                    m = (((1080 - y2) - (1080 - y1)) * 1.0) / (x2 - x1)
                    cRight = (1080 - y2) - (m * x2 * 1.0)
                    if (meanRight + 150) >= cRight >= (meanRight - 150):
                        outputRightLanes.append(rightLine)

            if flag:
                print "output right lanes: ", outputRightLanes
            if outputRightLanes == []:
                rightFlag = False

        if rightFlag:
            rightHistory = np.copy(outputRightLanes)
            for lane in outputRightLanes:
                for x1, y1, x2, y2 in lane:
                    cv2.line(img, (x1, y1), (x2, y2), [255, 0, 0], 10)

        if not rightFlag:

            if flag:
                print "using history here"
            for lane in rightHistory:
                for x1, y1, x2, y2 in lane:
                    cv2.line(img, (x1, y1), (x2, y2), [255, 255, 0], 11)

        return leftHistory, rightHistory







    @staticmethod
    def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, leftHistory, rightHistory,
                    minAngleLeftLanes, minAngleRightLanes,
                    maxAngleLeftLanes, maxAngleRightLanes, flag):
        """Returns an image with hough lines drawn"""
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                                maxLineGap=max_line_gap)
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        leftHistory, rightHistory = LaneFindingAlgorithm.draw_lines(line_img, lines, leftHistory, rightHistory,
                                                                    minAngleLeftLanes, minAngleRightLanes,
                                                                    maxAngleLeftLanes, maxAngleRightLanes, flag)
        return line_img, leftHistory, rightHistory










    @staticmethod
    def findLanes(input_image, rowROI, leftHistory, rightHistory, flag):
        """Performs operations on the input image using the rowROI coming in from pre processing
        :rtype: object  """
        with open('/home/ashwani/PycharmProjects/OpenCV/LaneDetectionResearchWork/params.tsv', 'rb') as f:
            reader = csv.reader(f, delimiter='=', quoting=csv.QUOTE_NONE)
            configuration = {}
            for row in reader:
                configuration[row[0]] = int(row[1])

        print '*************************************************************************************************'
        kernel_size = configuration['kernel_size']  # Gaussian blur kernel size
        low_threshold = configuration['low_threshold']  # Canny low threshold for gradient value
        high_threshold = configuration['high_threshold']  # Canny high threshold for gradient value
        rho = configuration['rho']  # distance resolution in pixels of the Hough grid
        theta = (np.pi / 180)  # angular resolution in radians of the Hough grid
        threshold = configuration['threshold']  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = configuration['min_line_length']  # minimum number of pixels making up a line
        max_line_gap = configuration['max_line_gap']  # maximum gap in pixels between connectible line segments

        minAngleLeftLanes = configuration['minAngleLeftLanes']
        minAngleRightLanes = configuration['minAngleRightLanes']
        maxAngleLeftLanes = configuration['maxAngleLeftLanes']
        maxAngleRightLanes = configuration['maxAngleRightLanes']

        # Defining the shape for masking
        left_bottom = [0, input_image.shape[0]]
        right_bottom = [input_image.shape[1], input_image.shape[0]]
        left_top = [0, rowROI]
        right_top = [input_image.shape[1], rowROI]

        if flag:
            # To show dotted lines around the ROI
            # x = [left_bottom[0], right_bottom[0], right_top[0], left_top[0], left_bottom[0]]
            # y = [left_bottom[1], right_bottom[1], right_top[1], left_top[1], left_bottom[1]]
            # plt.plot(x, y, '--', lw=2)
            plt.imshow(input_image)
            plt.show()

        test_image = np.copy(input_image)
        gray_image = LaneFindingAlgorithm.grayscale(test_image)
        # blur_gray = LaneFindingAlgorithm.gaussian_blur(gray_image, kernel_size)
        edges = LaneFindingAlgorithm.abs_sobel_thresh(gray_image, low_threshold, high_threshold)

        if flag:
            # plt.plot(x, y, '--', lw=2)
            plt.imshow(edges, cmap='gray')
            plt.show()

        vertices = np.array([[(left_bottom[0], left_bottom[1]), (left_top[0], left_top[1]),
                              (right_top[0], right_top[1]), (right_bottom[0], right_bottom[1])]], dtype=np.int32)
        masked_edges = LaneFindingAlgorithm.region_of_interest(edges, vertices)

        if flag:
            # plt.plot(x, y, '--', lw=2)
            plt.imshow(masked_edges, cmap='gray')
            plt.show()

        # Apply Hough transform to masked edge-detected image
        line_image, leftHistory, rightHistory = LaneFindingAlgorithm.hough_lines(masked_edges, rho, theta, threshold,
                                                                                 min_line_length,
                                                                                 max_line_gap, leftHistory,
                                                                                 rightHistory, minAngleLeftLanes,
                                                                                 minAngleRightLanes,
                                                                                 maxAngleLeftLanes, maxAngleRightLanes, flag)

        # Add line image to the input image
        # Explained in readme
        output_image = cv2.addWeighted(input_image, 0.8, line_image, 1, 0)

        if flag:
            plt.imshow(output_image)
            plt.show()
        return output_image, leftHistory, rightHistory
