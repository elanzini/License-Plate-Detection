import cv2
import os
import pandas as pd
import Localization
import Recognize
import Shot_Transition
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

THRESHOLD_SCENE = 0.9


def CaptureFrame_Process(file_path, sample_frequency, save_path):
    start_time = time.time()
    last_frame = None
    frame_count = 0
    cap = cv2.VideoCapture(file_path)

    while cap.isOpened():

        frame_count = frame_count + 1
        print("frame number: " + str(frame_count))

        ret, frame = cap.read()

        if frame is None:
            break

        if last_frame is None or Shot_Transition.get_histogram_correlation_grayscale(frame, last_frame) < THRESHOLD_SCENE:

            cv2.imshow("NEW SCENE", frame)
            plate_images = Localization.plate_detection(frame)

            for i in range(len(plate_images)):
                # cv2.imshow("Plate " + str(i), plate_images[i])
                end_time = time.time()
                print("Found after: " + str((end_time - start_time)))
                print("License Plate" + str(i) + ": " + Recognize.segment_and_recognize(plate_images[i]))
            cv2.waitKey()

            last_frame = frame

    cap.release()
    cv2.destroyAllWindows()


class Output:
    def __init__(self, frame_number, time_plate, plate_img=None, license_plate=None):
        self.plate_img = plate_img
        self.frame_number = frame_number
        self.time_plate = time_plate
        self.license_plate = license_plate