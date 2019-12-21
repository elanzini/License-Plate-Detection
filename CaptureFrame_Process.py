import cv2
import os
import pandas as pd
import numpy as np
import Localization
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
        ret, imgOriginalScene = cap.read()
        # imgLicensePlate = cv2.imread("lpd_test_01.png")
        # cv2.imshow("LP", imgLicensePlate)

        imgPossiblePlate = Localization.plate_detection(imgOriginalScene)
        if imgPossiblePlate is not None:
            cv2.imshow("Plate", imgPossiblePlate)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def get_cells(img_plate):
    (h, w) = img_plate.shape[:2]
    image_size = h * w
    mser = cv2.MSER_create()
    mser.setMaxArea(image_size // 2)
    mser.setMinArea(10)

    gray = cv2.cvtColor(img_plate, cv2.COLOR_BGR2GRAY)  # Converting to GrayScale
    _, bw = cv2.threshold(gray, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    regions, rects = mser.detectRegions(bw)

    # With the rects you can e.g. crop the letters
    for (x, y, w, h) in rects:
        if w * h > 600:
            cv2.rectangle(img_plate, (x, y), (x + w, y + h), color=(255, 0, 255), thickness=1)

    return img_plate