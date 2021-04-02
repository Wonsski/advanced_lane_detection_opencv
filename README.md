# advanced_lane_detection_opencv
This repository contains a Python implementation for lane detection using opencv. 

![Program output](screenshots/result.gif)

It uses computer vision techniques such as the sliding window algorithm preceded by image wrapping and thresholding.

![Image comparison](screenshots/thresholding-and-detection.gif)

# Requirements
- Python 3.x
- OpenCV
- Numpy
- Matplotlib

All requirements are in the requirements.txt file.

# Usage
```sh
python3 main.py
```
For changing video source just change this line in main.py
```sh
cap = cv2.VideoCapture('VIDEO_FILENAME.mp4')
```