import cv2
import numpy as np

def canny(image):
    # cvtColor(src, code, dst=None, dstCn=None)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny(image, threshold1, threshold2, edges=None, apertureSize=None, L2gradient=None)
    canny = cv2.Canny(blur, 50, 150)
    return canny

# Find coordinates from Slope
def region_of_intrest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height),(550, 250)]
    ])
    mask = np.zeros_like(image)
    # shading our region of intrest to white
    cv2.fillPoly(mask, polygons, 255)
     # bitwise with zeros image with intrested region
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        # print('Slope, intercepts ', parameters)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    print('left:', left_fit)
    print('right:', right_fit)
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

def display_lines(image, lines):
    lines_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            # print(line)
            # Converting from 2 dimen to 1 dminen
            x1, y1, x2, y2 = line.reshape(4)
            # lline(img, pt1, pt2, color, thickness=None, lineType=None, shift=None)
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    return lines_image
    

image = cv2.imread('lane_image.jpg')
print(image.shape)

cv2.imshow('Driving lane - Orginal', image)
cv2.waitKey(0)

lane_image = np.copy(image)
canny = canny(lane_image)
cv2.imshow('Driving lane - Grayed, Blured, Gradient, canny', canny)
cv2.waitKey(0)
cropped_image = region_of_intrest(canny)

cv2.imshow('Driving lane -Region of intrest ', cropped_image)
cv2.waitKey(0)

# Hough transform - Drawing line by voting using polar co-ordinates (p,@) - p = x cos@ + y sin@ = (y=mx+c)
# p - perpendicular line from (0,0), @ - angle of degree from x-axis

# HoughLinesP(image, rho, theta, threshold, lines=None, minLineLength=None, maxLineGap=None)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

averaged_lines = average_slope_intercept(image, lines)

line_image = display_lines(lane_image, averaged_lines)
# cv2.imshow('Driving lane', line_image)
# cv2.waitKey(0)

averaged_lines = average_slope_intercept(lane_image, lines)
# addWeighted(src1, alpha, src2, beta, gamma, dst=None, dtype=None)
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
cv2.imshow('Driving lane', combo_image)
cv2.waitKey(0)