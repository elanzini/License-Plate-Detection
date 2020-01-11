import numpy as np
import cv2

DEBUG = True

min_plate_width = 80
min_plate_height = 20

min_plate_area = min_plate_width * min_plate_height

white_sensitivity = 60
white_region_sensitivity = 100

kernel_2x2 = np.ones((2, 2), dtype="uint8")

kernel_20x10 = np.ones((20, 10), dtype="uint8")

color_filters = {
    'yellow': {
        'lower': np.array([15, 80, 80], dtype='uint8'),
        'upper': np.array([80, 255, 255], dtype='uint8')
    }
}

min_cc_aspect_ratio = 1.5

min_plate_aspect_ratio = (300 / 66) * 0.7

min_plate_color_match_ratio = 0.3


def apply_mask(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)


def locate_plates(image, plates_color='yellow'):

    # Converting to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Masking for color
    hsv_filter = color_filters[plates_color]

    color_mask = cv2.inRange(hsv_image, hsv_filter['lower'], hsv_filter['upper'])

    # Closing mask

    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel_20x10)

    plates = []

    # Getting connected components

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(color_mask, connectivity=8)

    for label in range(1, num_labels):

        h, w = (
            stats[label, cv2.CC_STAT_HEIGHT],
            stats[label, cv2.CC_STAT_WIDTH]
        )

        # Filtering by area
        if stats[label, cv2.CC_STAT_AREA] < min_plate_area:
            continue

        aspect_ratio = w / h

        # Filtering by connected component aspect ratio
        if aspect_ratio < min_cc_aspect_ratio:
            continue

        plate_points = np.argwhere(labels == label)

        plate_center, plate_dimensions, plate_angle = cv2.minAreaRect(plate_points)

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

        # Filtering by plate size

        if plate_width < min_plate_width:
            continue

        if plate_height < min_plate_height:
            continue

        y, x = (
            stats[label, cv2.CC_STAT_TOP],
            stats[label, cv2.CC_STAT_LEFT],
        )

        # Cropping plate region

        plate_region = image[y:y+h, x:x+w]

        # Rotating plate

        rotation_center = (plate_center[0] - x, plate_center[1] - y)
        rotation_matrix = cv2.getRotationMatrix2D(rotation_center, -plate_angle, 1)
        plate_region = cv2.warpAffine(plate_region, rotation_matrix, plate_region.shape[1::-1])

        # Cropping out plate

        half_plate_width = int(plate_width / 2)
        half_plate_height = int(plate_height / 2)

        plate_from_x = max(rotation_center[0] - half_plate_width, 0)
        plate_to_x = min(rotation_center[0] + half_plate_width, w - 1)

        plate_from_y = max(rotation_center[1] - half_plate_height, 0)
        plate_to_y = min(rotation_center[1] + half_plate_height, h - 1)

        plate = plate_region[plate_from_y:plate_to_y, plate_from_x:plate_to_x]

        # Filtering by color match ratio
        plate_hsv = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)
        plate_color_mask = cv2.inRange(plate_hsv, hsv_filter['lower'], hsv_filter['upper'])

        plate_color_match_ratio = cv2.countNonZero(plate_color_mask) / (plate_width * plate_height)

        if plate_color_match_ratio < min_plate_color_match_ratio:
            continue

        plates.append(plate)

    return plates_color, plates

