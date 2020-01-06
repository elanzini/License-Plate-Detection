import cv2
import numpy as np
import main

"""
In this file, you need to define plate_detection function.
To do:
    1. Localize the plates and crop the plates
    2. Adjust the cropped plate images
Inputs:(One)
    1. image: captured frame in CaptureFrame_Process.CaptureFrame_Process function
    type: Numpy array (imread by OpenCV package)
Outputs:(One)
    1. plate_imgs: cropped and adjusted plate images
    type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
Hints:
    1. You may need to define other functions, such as crop and adjust function
    2. You may need to define two ways for localizing plates(yellow or other colors)
"""

first_dilation_kernel = np.ones((10, 20), np.uint8)
first_dilation_iterations = 3

first_erosion_kernel = np.ones((3, 3), np.uint8)

hsv_filter_lower = np.array([20, 100, 0], dtype="uint8")
hsv_filter_upper = np.array([40, 255, 255], dtype="uint8")

hsv_filter_lower_component = np.array([20, 80, 0], dtype="uint8")
hsv_filter_upper_component = np.array([80, 255, 255], dtype="uint8")

color_match_ratio_threshold = 0.1

aspect_ratio_threshold = 1.5


def paste_component(canvas, component, message, location, color):

    canvas[location:location + component.shape[0], 0:component.shape[1]] = component
    cv2.putText(canvas, message, (component.shape[1], location + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)


def plate_detection(input_image):

    image = input_image
    height, width, channels = image.shape

    hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

    # HSV Filter
    mask = cv2.inRange(hsv_image, hsv_filter_lower, hsv_filter_upper)

    # Applying erosion
    mask = cv2.erode(mask, first_erosion_kernel, iterations=1)

    # Applying dilation
    mask = cv2.dilate(mask, first_dilation_kernel, iterations=first_dilation_iterations)

    # Getting connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)

    plates = []

    # Filtering connected components

    for label in range(num_labels):

        # Excluding black background

        if stats[label, cv2.CC_STAT_WIDTH] == width and stats[label, cv2.CC_STAT_HEIGHT] == height:
            continue

        aspect_ratio = stats[label, cv2.CC_STAT_WIDTH] / stats[label, cv2.CC_STAT_HEIGHT]

        # Filtering by aspect ratio

        if aspect_ratio < aspect_ratio_threshold:

            continue

        # Crop component

        x, y, w, h = cv2.boundingRect(np.argwhere(labels == label))

        cropped_component = image[x:x+w, y:y+h]

        if cropped_component.shape[0] > 0 and cropped_component.shape[1] > 0:

            yellow_mask = cv2.inRange(
                cropped_component,
                hsv_filter_lower_component,
                hsv_filter_upper_component
            )

            yellow_component_parts = cv2.countNonZero(yellow_mask)
            cropped_component_size = cropped_component.size / 3

            color_match_ratio = yellow_component_parts / cropped_component_size

            # Filtering by color matching ratio

            if color_match_ratio > color_match_ratio_threshold:

                # Extracting plate from plate region
                plate = plate_from_plate_region(cropped_component, yellow_mask)

                if plate is not None:
                    plates.append(plate)

    return plates


def plate_from_plate_region(plate_region, yellow_mask):

    # 2 x 2 dilation
    yellow_mask = cv2.dilate(yellow_mask, np.ones((2, 2), dtype="uint8"))

    # Connected components detection
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(yellow_mask, connectivity=4)

    # Getting connected component with max area

    max_connected_component_area = 0
    max_area_connected_component_label = -1

    for label in range(num_labels):

        component_area = stats[label, cv2.CC_STAT_AREA]
        component_size = stats[label, cv2.CC_STAT_WIDTH] * stats[label, cv2.CC_STAT_HEIGHT]

        if yellow_mask.size > component_size and component_area > max_connected_component_area:
            max_connected_component_area = component_area
            max_area_connected_component_label = label

    # Getting plate bounding tilted rectangle
    plate_center, plate_dimensions, plate_angle = cv2.minAreaRect(np.argwhere(labels == max_area_connected_component_label))

    plate_center = (int(plate_center[1]), int(plate_center[0]))

    if plate_dimensions[0] > plate_dimensions[1]:

        # Plate is tilted to the left

        plate_width = plate_dimensions[0]
        plate_height = plate_dimensions[1]

        plate_angle += 90

    else:

        # Plate is tilted to the right

        plate_width = plate_dimensions[1]
        plate_height = plate_dimensions[0]

    # Adjusting plate rotation

    rotation_matrix = cv2.getRotationMatrix2D(plate_center, -plate_angle, 1)
    plate_region = cv2.warpAffine(plate_region, rotation_matrix, plate_region.shape[1::-1])

    # Cropping out plate

    half_plate_width = max(int(plate_width / 2), 1)
    half_plate_height = max(int(plate_height / 2), 1)

    height, width, _ = plate_region.shape

    plate_from_x = max(plate_center[0] - half_plate_width, 0)
    plate_to_x = min(plate_center[0] + half_plate_width, width - 1)

    plate_from_y = max(plate_center[1] - half_plate_height, 0)
    plate_to_y = min(plate_center[1] + half_plate_height, height - 1)

    plate_region_cropped = plate_region[plate_from_y:plate_to_y, plate_from_x:plate_to_x]

    if main.DEBUG:
        cv2.imshow("test", plate_region_cropped)

    return plate_region_cropped
