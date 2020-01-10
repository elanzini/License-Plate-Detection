import cv2
import os
import re
import pandas as pd
import Localization
import Recognize
import Shot_Transition
import Validator
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
THRESHOLD_ECR = 0.15


def CaptureFrame_Process(file_path, sample_frequency, save_path):

    # Output csv
    df = pd.DataFrame(columns=['License Plate', 'Frame no.', 'Timestamp(seconds)'])

    # Time tracker + Frame Counter
    start_time = time.time()
    frame_count = 0
    plate_not_found = True
    scene_count = 0
    last_license_plate = None

    last_frame = None
    cap = cv2.VideoCapture(file_path)

    while cap.isOpened():

        frame_count = frame_count + 1

        ret, frame = cap.read()

        if frame is None:
            df.to_csv("outDetection.csv", encoding='utf-8', index=False)
            break

        if last_frame is None or \
                (Shot_Transition.get_histogram_correlation_grayscale(frame, last_frame) < THRESHOLD_SCENE or
                 Shot_Transition.ECR(frame, last_frame, frame.shape[1], frame.shape[0]) < THRESHOLD_ECR):

            plate_not_found = True
            # cv2.imshow("Scene", frame)
            # cv2.waitKey()

            plate_images = Localization.plate_detection(frame)

            for i in range(len(plate_images)):
                # Compute Time and License Plate
                ratio_plate = plate_images[i].shape[1]/plate_images[i].shape[0]
                if ratio_plate > 3.75:
                    license_plate = Recognize.segment_and_recognize(plate_images[i])
                    # cv2.imshow("Plate", plate_images[i])
                    # cv2.waitKey()
                    # cv2.imwrite("Plates/plate_" + str(frame_count) + "_" + str(i) + ".png", plate_images[i])
                    print("License Plate: " + license_plate)
                    if Validator.pattern_check_dutch_license(license_plate):
                        if last_license_plate is None or hamming_distance(last_license_plate, license_plate) > 3:
                            # print("License Plate: " + license_plate)
                            print("The plate is valid")
                            end_time = time.time()
                            time_to_compute = '%.3f' % (end_time - start_time)
                            df.loc[scene_count] = [license_plate] + [frame_count] + [1]
                            plate_not_found = False
                            last_license_plate = license_plate

                scene_count = scene_count + 1

            last_frame = frame

        elif plate_not_found:
            plate_images = Localization.plate_detection(frame)
            for i in range(len(plate_images)):
                # Compute Time and License Plate
                ratio_plate = plate_images[i].shape[1] / plate_images[i].shape[0]
                if ratio_plate > 3.75:
                    license_plate = Recognize.segment_and_recognize(plate_images[i])
                    # cv2.imshow("Plate", plate_images[i])
                    # cv2.waitKey()
                    # cv2.imwrite("Plates/plate_" + str(frame_count) + "_" + str(i) + ".png", plate_images[i])
                    print("License Plate: " + license_plate)
                    if Validator.pattern_check_dutch_license(license_plate):
                        if last_license_plate is None or hamming_distance(last_license_plate, license_plate) > 3:
                            # print("License Plate: " + license_plate)
                            print("The plate is valid")
                            end_time = time.time()
                            time_to_compute = '%.3f' % (end_time - start_time)
                            df.loc[scene_count] = [license_plate] + [frame_count] + [1]
                            plate_not_found = False
                            last_license_plate = license_plate

                scene_count = scene_count + 1

            last_frame = frame

    cap.release()
    cv2.destroyAllWindows()

def hamming_distance(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))