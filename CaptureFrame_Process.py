import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import Localization
import Preprocess
import Recognize

"""
In this file, you will define your own CaptureFrame_Process funtion. In this function,
you need three arguments: file_path(str type, the video file), sample_frequency(second), save_path(final results saving path).
To do:
	1. Capture the frames for the whole video by your sample_frequency, record the frame number and timestamp(seconds).
	2. Localize and recognize the plates in the frame.(Hints: need to use 'Localization.plate_detection' and 'Recognize.segmetn_and_recognize' functions)
	3. If recognizing any plates, save them into a .csv file.(Hints: may need to use 'pandas' package)
Inputs:(three)
	1. file_path: video path
	2. sample_frequency: second
	3. save_path: final .csv file path
Output: None
"""


def CaptureFrame_Process(file_path, sample_frequency, save_path):
    cap = cv2.VideoCapture(file_path)
    while (cap.isOpened()):

        ret, frame = cap.read()

        height, width, numChannels = frame.shape
        imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
        imgThreshScene = np.zeros((height, width, 1), np.uint8)
        imgContours = np.zeros((height, width, 3), np.uint8)

        imgGrayscaleScene, imgThreshScene = Preprocess.preprocess(frame)  # preprocess to get grayscale and threshold images

        cv2.imshow("gray", imgGrayscaleScene)
        cv2.imshow("threshold", imgThreshScene)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
#end function