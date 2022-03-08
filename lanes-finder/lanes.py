import os
import cv2
import numpy as np
#import matplotlib.pyplot as plt


def process_canny(image):

    # transfer the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # gaussian blur with 5*5 kernel, reduces noise (smoothing)
    # (not necessery if cv2.Canny() method would be later applied)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # cv2.Canny() computes the gradient in all directions of image
    # trace strongest gradients as a series of white pixels
    # thresholds allows to isolate the adjacent pixles that follow the strongest gradients
    # gradient > high_threshold --> accepted as edge pixel
    # gradient < low_threshold --> rejected
    # low_threshold < gradient < high_threshold --> accepted only if connected to a strong edge
    low_threshold = 50  #
    high_threshold = 150  # accepted
    canny = cv2.Canny(blur, low_threshold, high_threshold)

    return canny


# strong value of a pixle 255 represented in binary numbers 11111111 takes 8bit for 1 byte
def region_of_interest(image):
    # returns the enclosed region of field of view
    # recall the enclosed region was triangular in shape
    # the triangle be traced with vertices that go 200 to 1100 along the X
    # and vertically until the extent of the image (height from up to down)
    height = image.shape[0]
    # array of triangle(s), coordinates check by matplotlib image
    triangle = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    # set a black mask with same dimensions of the image
    mask = np.zeros_like(image)
    # fill the triangle area in the mask (white 255)
    cv2.fillPoly(mask, triangle, 255)
    # computing the bitwise & of both images
    # take the bitwise & of each homologous pixel in both arrays
    # ultimately masking the canny image to only show the region of interest
    # traced by the triangle contour of the mask
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image


def make_coordinates(image, line_parameters):
    # unpack the parameters
    slope, intercept = line_parameters
    # image size (704, 1279, 3): height, width, channel
    y1 = int(image.shape[0])
    # manuel set up the height of line
    y2 = int(y1*(3/5))
    # calculate the x
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)

    return np.array([x1, y1, x2, y2])


# average out the slope in y intercept into a single line
def average_slope_intercept(image, lines):
    # contain the coordinates of the average lines on the left
    left_fit = []
    # contain the coordinates of the average lines on the right
    right_fit = []
    # loop through every detected lines previously
    for line in lines:
        # unpack the line into 4 endpoints
        x1, y1, x2, y2 = line.reshape(4)
        # polyfit() fits a 1st degree polynomial which simply be a linear
        # function of y = m*x + b to the points and returns vector of coefficients
        # which describe the slope and y intercept (截距): [slope, y-intercept]
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        # caculate the slope (斜率)
        slope = parameters[0]
        intercept = parameters[1]
        # lines with positive slope is on right side (y-axis reversed in the image)
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    # everage out all the values of each side, axis=0 for vertical operating
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    # specify the coordinates of the both lines: x1, y1, x2, y2 for each lines
    # returned np.array([x1, y1, x2, y2])
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    averaged_lines = np.array([left_line, right_line])

    return averaged_lines


def display_lines(image, lines):
    # define a black image with the same size of target image
    black_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            # draw blue (255, 0, 0) lines on the black line_image with thickness of 10
            cv2.line(black_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    return black_image


'''
######################################## Image Processing ################################################
# load image
image = cv2.imread(
    os.getcwd() + '\\data\\image\\test_image.jpg'
)
# make a copy of the image
lane_image = np.copy(image)
# process canny function
canny_image = process_canny(lane_image)
# get the triangle area
cropped_region = region_of_interest(canny_image)
#####################################################
# Line Detection with Hough Transformation
# y = m*x + b
# Hough space: space of line coefficients (m, b), those all cross the certain dot (1, 8)
# (2, 12) --> 12=m*2+b --> b=-2*m+12
# (1, 8) --> 8=m*1+b --> b=-1*m+8
# cross point m=4, b=4
# coefficient of a line that crosses 2 dots, is the cross point (intersection) of their Hough spaces
# y = 4*x+4 crosses both (2, 12) and (1, 8)
#####################################################
# points, their Hough spaces have intersections locate in an tolerated area (bin), would be seen as a line
# cast a vote for every intersection inside of the bin, bin with max number of vote is the line of best fit
#####################################################
# PROBLEM: since a vertical line can not be represented in Hough space
# line would be written in Hesse normal form: r = x*cos(theta) + y*sin(theta), also: polar coordinates
# pi = x*cos(theta) + y*sin(theta)  --> sinusoidal curve in Hough space of theta and pi
# target: find the most voted intersection bin of the curves in Hough space
# for series of points in cartesian coordinate
#####################################################
# hough transform, size of the bins specified as 2 pixel, accompanied by a 1 degree precision
# 180°=pi radians, 1°=pi/180 radian, the smaller, the preciser
# min 100 intersections in a bin for it to be accepted as irrelevant line in describing the data
# an empty array as place holder, and length of a line in pixels that be accepted into the output in 40
# max distance in pixels between segmented lines, which alled to be connected
lines = cv2.HoughLinesP(
    cropped_region,
    2,
    np.pi/180,
    100,
    np.array([]),
    minLineLength=40,
    maxLineGap=5
)
#####################################################
# calculate the averaged lines of both sides
averaged_lines = average_slope_intercept(lane_image, lines)
# draw the detected lines on a black image with the same size of lane_image
line_image = display_lines(lane_image, averaged_lines)
# blend the lines to the original image
# adWeighted() takes the sum of the color image with line image
# any pixel value of original image add 0 (black) won't be changed, gamma=1
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
# 'result' is the showed window name
cv2.imshow('result', canny_image)
cv2.waitKey(0)

# imshow() receives an image only
#plt.imshow(canny)
#plt.show()
##########################################################################################################
'''


# capture video
cap = cv2.VideoCapture(
    os.getcwd() + '\\data\\video\\test2.mp4'
)
while (cap.isOpened()):
    # only the second value is a single frame (image)
    _, frame = cap.read()
    # same process for each frame
    canny_image = process_canny(frame)
    cropped_canny = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(
        cropped_canny,
        2,
        np.pi / 180,
        100,
        np.array([]),
        minLineLength=40,
        maxLineGap=5
    )
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('result', combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
