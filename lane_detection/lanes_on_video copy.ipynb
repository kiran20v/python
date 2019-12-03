import cv2
import numpy as np

# Canny image
def canny(image):
    # cvtColor(src, code, dst=None, dstCn=None)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny(image, threshold1, threshold2, edges=None, apertureSize=None, L2gradient=None)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_intrest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height),(550, 250)]        
    ], np.int32)
    
    mask = np.zeros_like(image)
    # shading our region of intrest to white
    cv2.fillPoly(mask, polygons, 255)
     # bitwise with zeros image with intrested region
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# Find coordinates from Slope
def make_points(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
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
    if len(left_fit) and len(right_fit):
    # add more weight to longer lines
        left_fit_average  = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        left_line  = make_points(image, left_fit_average)
        right_line = make_points(image, right_fit_average)
        averaged_lines = [left_line, right_line]
        return averaged_lines

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
    
cap = cv2.VideoCapture("driving_video.mp4")
while(cap.isOpened()):
    _,frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_intrest(canny_image)
    # Hough transform - Drawing line by voting using polar co-ordinates (p,@) - p = x cos@ + y sin@ = (y=mx+c)
    # p - perpendicular line from (0,0), @ - angle of degree from x-axis
    # HoughLinesP(image, rho, theta, threshold, lines=None, minLineLength=None, maxLineGap=None)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    # addWeighted(src1, alpha, src2, beta, gamma, dst=None, dtype=None)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('Driving lane', combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()