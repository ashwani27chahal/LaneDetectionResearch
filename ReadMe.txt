This is the write-up for Lane Detection Project.
The folder contains the following 4 files -

1. LaneFindingAlgorithm.py   -- contains the python code for finding lanes in an image
2. LaneDetectionAlgorithmTest.py -- contains the python code for reading images from a folder and testing LaneFindingAlgorithm.py
3. params.tsv --contains the values for different parameters used in the code 'LaneFindingAlgorithm.py '
4. Stats.txt -- contains the statistics for the 2 sets of images - 'Sunny data' and 'Rainy Data'

To execute the project, go to LaneDetectionAlgorithmTest.py and in last 4 lines of the code, provide the relevant values for -
  -> input_directory
  ->  output_directory
  ->  debug

And make a call to the method -
  ->  processing(input_directory, output_directory, debug)


There are two sets of data -

1. Sunny Data
---------------

Contains 6139 images.
Angle pramater values used -

minAngleLeftLanes=30
minAngleRightLanes=-70
maxAngleLeftLanes=70
maxAngleRightLanes=-30



1. Rainy Data
---------------

Contains 3800 images.
Angle pramater values used -

minAngleLeftLanes=25
minAngleRightLanes=-65
maxAngleLeftLanes=65
maxAngleRightLanes=-28



Algorithm Explained
---------------------

1. Image is read in LaneDetectionAlgorithmTest.py and is sent to LaneFindingAlgorithm.py (method findLanes)along with the y-axis rowI value which in this project is set to be the
   lower half of the image i.e. image.shape[0]/2  (dividing the y axis into half)

2. In LaneDetectionAlgorithmTest.py, in findLanes method, the image undergoes following pre processing -
        a) grayscaling --> LaneFindingAlgorithm.grayscale()
        b) gaussian blurring --> LaneFindingAlgorithm.gaussian_blur()

3. Grayscaled and blurred image is sent to a method LaneFindingAlgorithm.abs_sobel_thresh() which finds the horizontal gradience change (in the x-direction)
   and draws the edges on the image.

4. The image containing the edges, is masked using the ROI value (which was set as half of y-axis) in the method LaneFindingAlgorithm.region_of_interest()

5. The result of LaneFindingAlgorithm.region_of_interest() is a masked image containing edges. This image is passed through LaneFindingAlgorithm.hough_lines()
   which draws the lines for all the edges in the masked image and returns a line image containg lines in the desired region of interest.

6. Once this processing (step 1-5) is completed, method LaneFindingAlgorithm.draw_lines() is called. This method computes the slope for all the lines in the image.
   using formula  - angle = ((np.arctan(m)) / np.pi) * 180 where 'm' is the slope of line -> m = (((1080 - y2) - (1080 - y1)) * 1.0) / (x2 - x1)
   angle of each line with x-axis is computed.

   Angles are filtered using the parameters -
    minAngleLeftLanes=25
    minAngleRightLanes=-75
    maxAngleLeftLanes=75
    maxAngleRightLanes=-25

    Also, to filter out the extra lines with the same slope as of lane lines, y-intercept is calculated using the formula -
    c = (1080 - y2) - (m * x2 * 1.0) where m is the slope and (x2,y2) is any point on that line

    Using all the 'c' values for entire set of lines in left lane or y lane, those lines are filtered out which are either 150 pixels left or 150 pixels right
    to the mean value of 'c' : (meanLeft + 150) >= c >= (meanLeft - 150)

 7. To generate the output image, the line image is added to the input image using cv2.AddWeighted() method.
    cv2.AddWeighted(src1,alpha, src2, beta, gamma)
      src1 - first input aray (input image)
      alpha - weight of first array (0.8)
      src2 - second input array of same size and channel number (line_image)
      beta - weight of second array elements (1)
      gamma - scalar added to each sum (0)

8.  This output image along with the left lane and right lane history is returned and stored.



-------------------------------------

Find related work - and code (14 ,15 ,16 in pdf)