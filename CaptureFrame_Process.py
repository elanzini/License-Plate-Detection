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
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        plate_images = Localization.plate_detection(frame)
        for i in range(len(plate_images)):

            cv2.imshow("plate " + str(i), plate_images[i])
            print("plate " + str(i) + ": " + Recognize.segment_and_recognize(plate_images[i]))

    cap.release()
    cv2.destroyAllWindows()
