import cv2
import numpy as np

first_dilation_kernel = np.ones((10, 20), np.uint8)
first_dilation_iterations = 3

first_erosion_kernel = np.ones((2, 2), np.uint8)

region_color_match_ratio_threshold = 0.1

region_aspect_ratio_threshold = 1.5

region_dilation_kernel = np.ones((2, 2), dtype='uint8')

min_plate_width = 80
min_plate_height = 20

white_sensitivity = 60
white_region_sensitivity = 100

color_filters = {
    'yellow': {
        'lower': np.array([20, 100, 0], dtype='uint8'),
        'upper': np.array([40, 255, 255], dtype='uint8'),
        'region_lower': np.array([15, 100, 100], dtype='uint8'),
        'region_upper': np.array([80, 255, 255], dtype='uint8')
    },
    'white': {
        'lower': np.array([0, 0, 255 - white_sensitivity], dtype='uint8'),
        'upper': np.array([255, white_sensitivity, 255], dtype='uint8'),
        'region_lower': np.array([0, 0, 255 - white_region_sensitivity], dtype='uint8'),
        'region_upper': np.array([255, white_region_sensitivity, 255], dtype='uint8')
    }
}


possible_plate_colors = [
    'yellow',
    #'white'
]


def locate_plates(frame):

    for plate_color in possible_plate_colors:

        plates_by_color = locate_plate_by_color(frame, plate_color)

        if len(plates_by_color) > 0:

            return plate_color, plates_by_color

    return None, []


def locate_plate_by_color(input_image, color):

    image = input_image
    height, width, channels = image.shape

    hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

    # HSV Filter
    mask = cv2.inRange(hsv_image, color_filters[color]['lower'], color_filters[color]['upper'])

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

        if aspect_ratio < region_aspect_ratio_threshold:
            continue

        y, x, h, w = (
            stats[label, cv2.CC_STAT_TOP],
            stats[label, cv2.CC_STAT_LEFT],
            stats[label, cv2.CC_STAT_HEIGHT],
            stats[label, cv2.CC_STAT_WIDTH],
        )

        if h > 0 and w > 0:

            plate_color_mask = cv2.inRange(
                hsv_image[y:y+h, x:x+w],
                color_filters[color]['region_lower'],
                color_filters[color]['region_upper']
            )

            matching_color_component_parts = cv2.countNonZero(plate_color_mask)
            cropped_component_size = h * w

            color_match_ratio = matching_color_component_parts / cropped_component_size

            # Filtering by color matching ratio

            if color_match_ratio > region_color_match_ratio_threshold:

                # Extracting plate from plate region
                plate = plate_from_plate_region(image[y:y + h, x:x + w], plate_color_mask)

                if plate is not None:
                    plates.append(plate)
                    # cv2.imshow("plate", plate)

    return plates


def plate_from_plate_region(plate_region, plate_color_mask):

    # Dilation
    plate_color_mask = cv2.dilate(plate_color_mask, region_dilation_kernel)

    # Connected components detection
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(plate_color_mask, connectivity=4)

    # Getting connected component with max area

    max_connected_component_area = 0
    max_area_connected_component_label = -1

    for label in range(num_labels):

        component_area = stats[label, cv2.CC_STAT_AREA]
        component_size = stats[label, cv2.CC_STAT_WIDTH] * stats[label, cv2.CC_STAT_HEIGHT]

        if plate_color_mask.size > component_size and component_area > max_connected_component_area:
            max_connected_component_area = component_area
            max_area_connected_component_label = label

    # Getting plate bounding tilted rectangle
    plate_points = np.argwhere(labels == max_area_connected_component_label)
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

        return None

    if plate_height < min_plate_height:

        return None

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

    return plate_region_cropped

