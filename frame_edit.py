import cv2
import numpy as np

# Wrapping perspective
def wrapPerspective(image, roi):
    img_size = image.shape[::-1][1:]
    offset = 300

    dst = np.float32([
        [offset, 0], 
        [offset, img_size[1]],
        [img_size[0]-offset, img_size[1]], 
        [img_size[0]-offset, 0]
    ])

    matrix = cv2.getPerspectiveTransform(roi, dst)
    matrix_inverted = cv2.getPerspectiveTransform(dst, roi)
    wrapped_img = cv2.warpPerspective(image, matrix, img_size, flags=cv2.INTER_LINEAR) 
    
    return matrix_inverted,wrapped_img

def createMask(image):
    # hls image
    image = cv2.cvtColor(image,cv2.COLOR_BGR2HLS)
    
    # adding blur
    blur = cv2.GaussianBlur(image,(5, 5),0)

    # white color mask
    lower = np.uint8([0,200, 0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(blur, lower, upper)
    
    # yellow color mask
    lower = np.uint8([10, 0, 100])
    upper = np.uint8([25, 185, 255])
    yellow_mask = cv2.inRange(blur, lower, upper)

    # combining masks
    mask = cv2.bitwise_or(white_mask, yellow_mask)

    # dilation and erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    eroded = cv2.erode(mask,kernel)
    dilated = cv2.dilate(eroded,kernel)

    return dilated

def imageStacking(frame, wrapped_img, sliding_windows, lines, projected_image):

    frame = cv2.resize(frame, (int(frame.shape[1] * 25/100),int(frame.shape[0] * 25/100)))
    wrapped_img = cv2.resize(wrapped_img, (int(wrapped_img.shape[1] * 25/100),int(wrapped_img.shape[0] * 25/100)))
    sliding_windows = cv2.resize(sliding_windows, (int(sliding_windows.shape[1] * 25/100),int(sliding_windows.shape[0] * 25/100)))
    lines = cv2.resize(lines, (int(lines.shape[1] * 25/100),int(lines.shape[0] * 25/100)))

    right_side = np.vstack([frame,wrapped_img,sliding_windows,lines])
    result = np.hstack([projected_image,right_side])

    return result