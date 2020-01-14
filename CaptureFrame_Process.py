import cv2
import pandas as pd
import Shot_Transition
import Validator
import time
import PlateRecognition
import Localization
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
THRESHOLD_ECR = 0.3

DEBUG = False


def hamming_distance(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def get_most_common_plate_in_scene(scene):

    max_count = 0
    most_common_plate = '!'

    for plate in scene['plates']:
        if scene['plates'][plate] > max_count:
            max_count = scene['plates'][plate]
            most_common_plate = plate

    return most_common_plate


def join_scenes(scene0, scene1):

    plates = scene0['plates'].copy()

    for plate in scene1['plates']:

        if plate in plates:
            plates.update({
               plate: plates[plate] + scene1['plates'][plate]
            })
        else:
            plates.update({
                plate: scene1['plates'][plate]
            })

    return {
        'first_frame': min(scene0['first_frame'], scene1['first_frame']),
        'frame_counter': scene0['frame_counter'] + scene1['frame_counter'],
        'plates': plates
    }


def is_new_scene(previous_frame, current_frame):
    return (Shot_Transition.get_histogram_correlation_grayscale(current_frame, previous_frame) < THRESHOLD_SCENE or
                 Shot_Transition.ECR(current_frame, previous_frame, current_frame.shape[1], current_frame.shape[0]) < THRESHOLD_ECR)


def get_time_to_compute(start_time, current_time):

    return '%.3f' % (current_time - start_time)


def merge_similar_plates(plates_dict, max_hamming_distance):

    n = len(plates_dict)

    plates = list(plates_dict.keys())

    plates = sorted(plates, key=lambda plate: -plates_dict[plate])

    plates_to_join = []

    for i in range(n - 1):
        for j in range(i + 1, n):

            plate_i = plates[i]
            plate_j = plates[j]

            if hamming_distance(plate_i, plate_j) <= max_hamming_distance:

                plates_to_join.append(plate_i)
                plates_to_join.append(plate_j)

            if len(plates_to_join) > 0:
                break

        if len(plates_to_join) > 0:
            break

    if len(plates_to_join) > 0:

        new_plates_dict = plates_dict.copy()

        if plates_dict[plates_to_join[0]] > plates_dict[plates_to_join[1]]:

            del new_plates_dict[plates_to_join[0]]
            del new_plates_dict[plates_to_join[1]]

            new_plates_dict.update({
                plates_to_join[0]: plates_dict[plates_to_join[0]] + plates_dict[plates_to_join[1]]
            })

        else:

            del new_plates_dict[plates_to_join[0]]
            del new_plates_dict[plates_to_join[1]]

            new_plates_dict.update({
                plates_to_join[1]: plates_dict[plates_to_join[0]] + plates_dict[plates_to_join[1]]
            })

        return merge_similar_plates(new_plates_dict, max_hamming_distance)

    else:

        return plates_dict

def CaptureFrame_Process(file_path, sample_frequency, save_path):

    cap = cv2.VideoCapture(file_path)

    ret, frame = cap.read()
    previous_frame = frame
    total_frames_counter = 1
    scenes = []
    scene = {
        'first_frame': total_frames_counter,
        'frame_counter': 0,
        'plates': {}
    }

    plates_time_to_compute = {}

    # Output csv
    df = pd.DataFrame(columns=['License plate', 'Frame no.', 'Timestamp(seconds)'])
    start_time = time.time()

    if DEBUG:
        prev_plates_count = 0

    while cap.isOpened():

        previous_frame = frame
        ret, frame = cap.read()

        if frame is None:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        total_frames_counter += 1

        plates_color, plates = Localization.find_plates(frame)

        cv2.imshow("frame", frame)

        if is_new_scene(previous_frame, frame):

            if DEBUG:
                cv2.imshow("new scene frame", frame)

            scenes.append(scene.copy())
            scene = {
                'first_frame': total_frames_counter,
                'frame_counter': 0,
                'plates': {}
            }
        else:
            scene.update({
                'frame_counter': scene['frame_counter'] + 1
            })

        if DEBUG:

            for i in range(len(plates)):
                cv2.imshow("plate " + str(i), plates[i])

            if len(plates) < prev_plates_count:
                for i in range(len(plates), prev_plates_count + 1):
                    cv2.destroyWindow("plate " + str(i))

            prev_plates_count = len(plates)


        for i, plate in enumerate(plates):

            recognized_plate = PlateRecognition.recognize_plate(plate, plates_color)

            if recognized_plate is None:
                continue
            elif (plates_color in ['yellow', 'yellow_red_image'] and not Validator.pattern_check_dutch_license(recognized_plate)):
                continue

            if recognized_plate not in plates_time_to_compute:
                plates_time_to_compute.update({
                    recognized_plate: get_time_to_compute(start_time, time.time())
                })

            if recognized_plate in scene['plates']:

                scene['plates'].update({
                    recognized_plate: scene['plates'][recognized_plate] + 1
                })

            else:

                scene['plates'].update({
                    recognized_plate: 1
                })

    scenes.append(scene)

    if DEBUG:
        for scene in scenes:
            print(scene)

        print("\n")

    scenes = [scene for scene in scenes if scene['frame_counter'] > 0 and len(scene['plates']) > 0]

    if DEBUG:
        for scene in scenes:
            print(scene)

        print("\n")

    scenes_joined = []

    current_joined_scene = scenes[0]
    previous_scene_most_common_plate = get_most_common_plate_in_scene(scenes[0])

    for scene_index in range(1, len(scenes)):

        current_scene = scenes[scene_index]

        current_scene_most_common_plate = get_most_common_plate_in_scene(current_scene)

        if hamming_distance(current_scene_most_common_plate, previous_scene_most_common_plate) > 3:
            scenes_joined.append(current_joined_scene.copy())
            current_joined_scene = current_scene
            previous_scene_most_common_plate = get_most_common_plate_in_scene(current_joined_scene)
        else:
            current_joined_scene = join_scenes(current_joined_scene, current_scene)

    scenes_joined.append(current_joined_scene)

    if DEBUG:
        for scene in scenes_joined:
            print(scene)
        print('\n')

    for scene in scenes_joined:

        scene.update({
            'plates': merge_similar_plates(scene['plates'], 3)
        })

    if DEBUG:
        for scene in scenes_joined:
            print(scene)
        print('\n')

    plates_counter = 0

    for scene in scenes_joined:

        if len(scene['plates']) > 1:

            total_scene_plates = sum(scene['plates'].values())

            for plate in scene['plates']:

                plate_matches_count = scene['plates'][plate]

                if plate_matches_count / total_scene_plates >= 0.2:
                    df.loc[plates_counter] = [plate] + [(scene['first_frame'] + scene['frame_counter'] // 2)] + [plates_time_to_compute[plate]]
                    plates_counter += 1

        else:

            plate = get_most_common_plate_in_scene(scene)

            df.loc[plates_counter] = [plate] + [(scene['first_frame'] + scene['frame_counter'] // 2)] + [plates_time_to_compute[plate]]
            plates_counter += 1

    df.to_csv("out.csv", encoding="utf-8", index=False)

    cap.release()
    cv2.destroyAllWindows()