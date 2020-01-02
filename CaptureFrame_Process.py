import cv2
import os
import pandas as pd
import Localization
import Recognize
import time

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

    while cap.isOpened():

        ret, frame = cap.read()
        # Converts each frame to gray scale

        cv2.imshow('frame', frame)

        plate = Localization.plate_detection(frame)

        # Testing Canny Edge detector
        # cv2.imwrite('fame.png', gray)
        # cv2.imwrite('edges.png', edges)

        time.sleep(.05)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
