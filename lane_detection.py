import cv2
import numpy as np
import matplotlib.pyplot as plt

n_windows = 10
margin = 100
threshold = 40

def showHistogram(mask):
    histogram = np.sum(mask[int(mask.shape[0]/2):,:], axis=0)
    plt.plot(histogram)
    plt.show()

def getHistogramPeaks(histo):
    center = np.int(histo.shape[0]/2)
    left_x = np.argmax(histo[:center])
    right_x = np.argmax(histo[center:]) + center

    return left_x, right_x

def slidingWindows(mask, left_x, right_x, n_windows, margin, minpix):
    # Creating output 3-channel image
    sliding_windows = np.dstack((mask, mask, mask))*255

    # Sliding window size
    window_size = mask.shape[0]//n_windows
    
    # Identifying nonzeros
    nonzeros = mask.nonzero()
    nonzeros_x = nonzeros[1]
    nonzeros_y = nonzeros[0]

    curr_left_x = left_x
    curr_right_x = right_x

    left_fits = []
    right_fits = []

    for window in range(n_windows):
        # Window location
        # y
        win_y_top = mask.shape[0] - window * window_size
        win_y_bottom = mask.shape[0] - (window+1) * window_size

        # x
        win_left_x_min = curr_left_x - margin
        win_left_x_max = curr_left_x + margin

        win_right_x_min = curr_right_x - margin
        win_right_x_max = curr_right_x + margin

        # Drawing rectangles for left and right curve
        cv2.rectangle(sliding_windows, (win_left_x_min, win_y_bottom), (win_left_x_max, win_y_top), (255,0,0),2)
        cv2.rectangle(sliding_windows, (win_right_x_min, win_y_bottom), (win_right_x_max, win_y_top), (0,0,255),2)

        # Window fit
        leftFitting = ((nonzeros_y >= win_y_bottom) & (nonzeros_y < win_y_top) & (nonzeros_x >= win_left_x_min) & (nonzeros_x < win_left_x_max)).nonzero()[0]
        rightFitting = ((nonzeros_y >= win_y_bottom) & (nonzeros_y < win_y_top) & (nonzeros_x >= win_right_x_min) & (nonzeros_x < win_right_x_max)).nonzero()[0]
        
        # Adding fits to the list
        left_fits.append(leftFitting)
        right_fits.append(rightFitting)

        # Recenter future window if fits
        if len(leftFitting) > threshold:
            curr_left_x = np.int(np.mean(nonzeros_x[leftFitting]))
        if len(rightFitting) > threshold:
            curr_right_x = np.int(np.mean(nonzeros_x[rightFitting]))

    # Concat lists
    left_fits = np.concatenate(left_fits)
    right_fits = np.concatenate(right_fits)

    return left_fits, right_fits, nonzeros_x, nonzeros_y, sliding_windows

def genLinePoints(fitx, ploty):
    first_line_window = np.array([np.transpose(np.vstack([fitx-margin, ploty]))])
    second_line_window = np.array([np.flipud(np.transpose(np.vstack([fitx+margin, ploty])))])
    return np.hstack((first_line_window, second_line_window))

def drawCurves(mask, left_fitx, right_fitx, ploty, nonzeros_x, nonzeros_y, left_fits, right_fits):
    # Create blank images
    lines_img = np.dstack((mask, mask, mask))*255
    mask_img = np.zeros_like(lines_img)

    # Color mask curves
    mask_img[nonzeros_y[left_fits], nonzeros_x[left_fits]] = [255, 0, 0]
    mask_img[nonzeros_y[right_fits], nonzeros_x[right_fits]] = [0, 0, 255]

    # Getting line points
    left_points = genLinePoints(left_fitx, ploty)
    right_points = genLinePoints(right_fitx, ploty)

    # Draw polygons
    cv2.fillPoly(lines_img, np.int_([left_points]), (0,255, 0))
    cv2.fillPoly(lines_img, np.int_([right_points]), (0,255, 0))

    lines = cv2.addWeighted(lines_img, 0.25, mask_img, 1, 0)

    return lines

def laneProjection(frame, mask, ploty, left_fitx, right_fitx, matrix_inverted):
    # Blank 3 channel image
    blank_img = np.zeros_like(mask).astype(np.uint8)
    color_warp = np.dstack((blank_img, blank_img, blank_img))
    
    # Getting lane points
    points_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    points_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    points = np.hstack((points_left, points_right))
    
    # Drawing polygon
    cv2.fillPoly(color_warp, np.int_([points]), (0,255, 0))
    
    # Reverse warp perspective
    newwarp = cv2.warpPerspective(color_warp, matrix_inverted, (frame.shape[1], frame.shape[0]))
    
    # Combining lane with original frame
    projected = cv2.addWeighted(frame, 1, newwarp, 0.3, 0)
    return projected
    

def locateLane(mask, frame, matrix_inverted):
    try:
        histogram = np.sum(mask[int(mask.shape[0]/2):,:], axis=0)

        # Get the X values of the curves
        left_x_location, right_x_location = getHistogramPeaks(histogram)

        # Sliding windows
        left_fits, right_fits, nonzeros_x, nonzeros_y, sliding_windows = slidingWindows(mask, left_x_location, right_x_location, n_windows, margin, threshold)

        # Color mask curves
        sliding_windows[nonzeros_y[left_fits], nonzeros_x[left_fits]] = [255, 0, 0]
        sliding_windows[nonzeros_y[right_fits], nonzeros_x[right_fits]] = [0, 0, 255]

        # Sliding windows preview
        #cv2.imshow('Sliding windows',sliding_windows)

        # Fitting polunomials
        left_fit = np.polyfit(nonzeros_y[left_fits], nonzeros_x[left_fits], 2)
        right_fit = np.polyfit(nonzeros_y[right_fits], nonzeros_x[right_fits], 2)

        ploty = np.linspace(0, mask.shape[0]-1, mask.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        lines = drawCurves(mask,left_fitx,right_fitx,ploty,nonzeros_x,nonzeros_y, left_fits, right_fits)
        # Lines preview
        #cv2.imshow('Lines preview',lines)

        projected_lane = laneProjection(frame, mask, ploty, left_fitx, right_fitx, matrix_inverted)

        return sliding_windows, lines, projected_lane

    except:
        pass