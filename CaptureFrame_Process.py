import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
import Localization
import Preprocess
import Recognize
import product

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

        ret, imgOriginalScene = cap.read()

        height, width, numChannels = imgOriginalScene.shape
        imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
        imgThreshScene = np.zeros((height, width, 1), np.uint8)
        imgContours = np.zeros((height, width, 3), np.uint8)

        imgGrayscaleScene, imgThreshScene = Preprocess.preprocess(imgOriginalScene)  # preprocess to get grayscale and threshold images

        cv2.imshow("gray", imgGrayscaleScene)
        # LOCALIZATION MADE WITH YELLOW TRACKER IS TEMPORARY SINCE IT WILL NOT WORK with other LP
        # frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # frame_threshold = cv2.inRange(frame_HSV, (20, 100, 100), (30, 255, 255))
        # cv2.imshow("Yellow tracker", frame_threshold)

        listOfPossiblePlates = Localization.detectPlatesInScene(imgOriginalScene)
        if len(listOfPossiblePlates) == 0:
            print("\nno license plates were detected\n")
        else:
            listOfPossiblePlates.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)
            licPlate = listOfPossiblePlates[0]
            cv2.imshow("imgPlate", licPlate.imgPlate)
            cv2.imshow("imgThresh", licPlate.imgThresh)
            if len(licPlate.strChars) == 0:
                print("\nno characters were detected\n\n")
                return
            drawRedRectangleAroundPlate(imgOriginalScene, licPlate)
            print("\nlicense plate read from image = " + licPlate.strChars + "\n")
            print("----------------------------------------")
            product.writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)
            cv2.imshow("imgOriginalScene", imgOriginalScene)
            cv2.imwrite("imgOriginalScene.png", imgOriginalScene)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
#end function

def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):
    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), product.SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), product.SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), product.SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), product.SCALAR_RED, 2)