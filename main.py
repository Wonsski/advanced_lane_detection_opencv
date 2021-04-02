import cv2
import numpy as np
import frame_edit
import lane_detection

def main():
    cap = cv2.VideoCapture('project_video.mp4')

    roi = np.float32([
        (580, 450), # Top-left 
        (275, 675), # Bottom-left 
        (1130, 675), # Bottom-right 
        (690, 450) # Top-right 
    ])

    # Video play
    while cap.isOpened():
        _, frame = cap.read()

        # Perspective wrap
        matrix_inverted, wrapped_img = frame_edit.wrapPerspective(frame, roi)

        # Creating mask
        mask = frame_edit.createMask(wrapped_img)

        # Show pixel histogram
        #lane_detection.showHistogram(mask)

        # Detecting and visualising lanes
        try:
            sliding_windows, lines, projected_lane = lane_detection.locateLane(mask, frame, matrix_inverted)
        except:
            sliding_windows = wrapped_img
            lines = wrapped_img
            projected_lane = frame

        # Adding polylines
        cv2.polylines(frame, [roi.astype(int)], True, (0,255,0), 2)

        # Stacking all images
        result = frame_edit.imageStacking(frame, wrapped_img, sliding_windows, lines, projected_lane)
        cv2.imshow('Result', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

if __name__ == '__main__':
    main()