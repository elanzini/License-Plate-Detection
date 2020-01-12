import cv2
import os
import pandas as pd
import Shot_Transition
import LightPlateRecognition
import LightLocalization
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


frame_count = 0
plate_not_found = True
scene_count = 0
last_license_plate = None
last_frame = None
plate_not_found = True
start_time = 0

# Output csv
df = pd.DataFrame(columns=['License plate', 'Frame no.', 'Timestamp(seconds)'])


def hamming_distance(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def parse_frame(frame):

    global last_license_plate
    global scene_count
    global last_frame
    global plate_not_found
    global start_time
    global df

    plate_color, plate_images = LightLocalization.locate_plates(frame)
    for i in range(len(plate_images)):
        # cv2.imshow("Plate", plate_images[i])
        # cv2.waitKey()

        # Compute Time and License Plate
        ratio_plate = plate_images[i].shape[1] / plate_images[i].shape[0]
        license_plate = LightPlateRecognition.recognize_plate(plate_images[i], plate_color)
        # cv2.imwrite("Plates/plate_" + str(frame_count) + "_" + str(i) + ".png", plate_images[i])
        print("License Plate: " + license_plate)

        if (plate_color is 'yellow' and Validator.pattern_check_dutch_license(license_plate)) or \
                (plate_color is not 'yellow'):
            if last_license_plate is None or hamming_distance(last_license_plate, license_plate) > 3:
                # print("License Plate: " + license_plate)
                print("The plate is valid")
                end_time = time.time()
                time_to_compute = '%.3f' % (end_time - start_time)
                df.loc[scene_count] = [license_plate] + [frame_count] + [time_to_compute]
                plate_not_found = False
                last_license_plate = license_plate

        scene_count += 1

    last_frame = frame


def CaptureFrame_Process(file_path, sample_frequency, save_path):

    global frame_count
    global plate_not_found
    global scene_count
    global last_license_plate
    global start_time
    global df

    cap = cv2.VideoCapture(file_path)
    start_time = time.time()

    while cap.isOpened():

        frame_count += 1

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

            parse_frame(frame)

        elif plate_not_found:

            parse_frame(frame)

    cap.release()
    cv2.destroyAllWindows()