import cv2
import os
import pandas as pd
import Localization
import Recognize
import SceneDetection
import numpy as np

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

    nOfFramesApprox = int(cv2.VideoCapture.get(cap, int(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(cap.get(3))
    height = int(cap.get(4))

    ret, frame = cap.read()

    sceneDetector = SceneDetection.SceneDetector(width, height, nOfFramesApprox)

    while (cap.isOpened()):
        ret, frame = cap.read()
        # Converts each frame to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Applies Canny Edge Detector
        edges = cv2.Canny(gray, 100, 200)
        cv2.imshow('frame', gray)
        cv2.imshow('edges', edges)

        # Testing Canny Edge detector
        #cv2.imwrite('fame.png', gray)
        #cv2.imwrite('edges.png', edges)

        sceneDetector.captureFrame(gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    sceneDetector.getSceneChanges(35)

    cap.release()
    cv2.destroyAllWindows()
