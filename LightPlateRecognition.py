import numpy as np
import cv2

aspect_ratio_threshold = 3

letter_lower_limit = 0
letter_upper_limit = 110

centroid_y_threshold = .2
letter_component_max_width_ratio = 1 / 4
letter_component_height_min_ratio = 1 / 2

expected_number_of_letters = 5

kernel_2x2 = np.ones((2, 2), dtype="uint8")
kernel_3x3 = np.ones((2, 2), dtype="uint8")

letter_open_kernel = kernel_2x2

max_expected_letters = 6

DEBUG = False
debug_plate_letters_image_colors = np.array([
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255]
], dtype="uint8")

histogram_bins_width = 5
histogram_bins = np.arange(0, 255, histogram_bins_width)


def load_letter_templates():
    raw_templates = [
        [cv2.imread("SameSizeLetters/1.bmp"), "B"],
        [cv2.imread("SameSizeLetters/2.bmp"), "D"],
        [cv2.imread("SameSizeLetters/3.bmp"), "F"],
        [cv2.imread("SameSizeLetters/4.bmp"), "G"],
        [cv2.imread("SameSizeLetters/5.bmp"), "H"],
        [cv2.imread("SameSizeLetters/6.bmp"), "J"],
        [cv2.imread("SameSizeLetters/7.bmp"), "K"],
        [cv2.imread("SameSizeLetters/8.bmp"), "L"],
        [cv2.imread("SameSizeLetters/9.bmp"), "M"],
        [cv2.imread("SameSizeLetters/10.bmp"), "N"],
        [cv2.imread("SameSizeLetters/11.bmp"), "P"],
        [cv2.imread("SameSizeLetters/12.bmp"), "R"],
        [cv2.imread("SameSizeLetters/13.bmp"), "S"],
        [cv2.imread("SameSizeLetters/14.bmp"), "T"],
        [cv2.imread("SameSizeLetters/15.bmp"), "V"],
        [cv2.imread("SameSizeLetters/16.bmp"), "X"],
        [cv2.imread("SameSizeLetters/17.bmp"), "Z"],

        [cv2.imread("SameSizeNumbers/0.bmp"), "0"],
        [cv2.imread("SameSizeNumbers/1.bmp"), "1"],
        [cv2.imread("SameSizeNumbers/2.bmp"), "2"],
        [cv2.imread("SameSizeNumbers/3.bmp"), "3"],
        [cv2.imread("SameSizeNumbers/4.bmp"), "4"],
        [cv2.imread("SameSizeNumbers/5.bmp"), "5"],
        [cv2.imread("SameSizeNumbers/6.bmp"), "6"],
        [cv2.imread("SameSizeNumbers/7.bmp"), "7"],
        [cv2.imread("SameSizeNumbers/8.bmp"), "8"],
        [cv2.imread("SameSizeNumbers/9.bmp"), "9"],
    ]

    result = []

    for letter_template_image_raw, letter in raw_templates:
        raw_height = letter_template_image_raw.shape[0]

        letter_template_image_raw = cv2.cvtColor(letter_template_image_raw, cv2.COLOR_BGR2GRAY)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(letter_template_image_raw)

        assert num_labels == 2

        white_label = 1

        letter_width = stats[white_label, cv2.CC_STAT_WIDTH]

        result.append({
            raw_height: {
                "image": letter_template_image_raw[:, :letter_width],
                "center": (int(raw_height / 2), int(letter_width / 2)),
                "size": (raw_height, letter_width)
            },
            "letter": letter,
            "default_height": raw_height

        })

    return result


def resize_templates(templates, height):
    for template in templates:

        if height not in template:
            default_height = template['default_height']
            default_template_image = template[default_height]["image"]

            default_width = template[default_height]["size"][1]

            width = int(default_width * height / default_height)

            size = (width, height)

            template.update({
                height: {
                    "image": cv2.resize(default_template_image, dsize=size, interpolation=cv2.INTER_NEAREST),
                }
            })


letter_templates = load_letter_templates()


def count_mismatches_same_size(image1, image2):
    count = 0

    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            if image1[i, j] != image2[i, j]:
                count += 1

    return count


def count_mismatches_different_size(min_width_image, max_width_image):
    min_width = min_width_image.shape[1]
    max_width = max_width_image.shape[1]

    height = min_width_image.shape[0]

    left_padding = int((max_width - min_width) / 2)

    right_padding = max_width - min_width - left_padding

    mismatches = 0

    # Left padding region
    for y in range(height):
        for x in range(left_padding):
            if max_width_image[y, x] > 0:
                mismatches += 1

    # Right padding region

    for y in range(height):
        for x in range(left_padding + min_width, max_width):
            if max_width_image[y, x] > 0:
                mismatches += 1

    # Central part
    for y in range(height):
        for x in range(min_width):
            if min_width_image[y, x] != max_width_image[y, x + right_padding]:
                mismatches += 1

    return mismatches


def recognize_plate(plate, plate_color):
    height, width, _ = plate.shape
    plate_grayscale = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    # Detecting edges

    edges = cv2.Canny(plate, 100, 200)

    # Dilating edges

    edges = cv2.dilate(edges, kernel_2x2)

    if DEBUG:
        cv2.imshow("edges", edges)

    # Blob detection
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity=8)

    centroid_lower_y_limit = int(height / 2 - centroid_y_threshold * height)
    centroid_upper_y_limit = int(height / 2 + centroid_y_threshold * height)

    min_component_height = int(letter_component_height_min_ratio * height)

    max_component_width = width * letter_component_max_width_ratio

    letters = []

    for label in range(num_labels):

        # Filtering by centroid position
        if centroids[label][1] < centroid_lower_y_limit or centroids[label][1] > centroid_upper_y_limit:
            continue

        letter_width = stats[label, cv2.CC_STAT_WIDTH]

        # Filtering by component width
        if letter_width > max_component_width:
            continue

        letter_height = stats[label, cv2.CC_STAT_HEIGHT]

        # Filtering by component height
        if letter_height < min_component_height:
            continue

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]

        letter_grayscale = plate_grayscale[y:y + letter_height, x:x + letter_width]

        letter_grayscale_histogram = np.histogram(letter_grayscale, bins=histogram_bins)[0]

        avg = np.average(letter_grayscale)

        last_black_bin_index = int(np.floor(avg / histogram_bins_width))

        main_black_bin_index = np.argmax(letter_grayscale_histogram[:last_black_bin_index])
        main_yellow_bin_index = np.argmax(letter_grayscale_histogram[last_black_bin_index + 1:]) + last_black_bin_index

        threshold = int((main_yellow_bin_index + main_black_bin_index) / 2) * histogram_bins_width

        letter_mask = np.zeros(letter_grayscale.shape, dtype="uint8")
        letter_mask[np.where(letter_grayscale < threshold)] = 255

        letter_mask = cv2.morphologyEx(letter_mask, cv2.MORPH_OPEN, letter_open_kernel)

        letters.append({
            'position': x,
            'letter_mask': letter_mask
        })

    licence_plate_letters = []

    if DEBUG:
        debug_plate_letters_image = np.zeros(plate.shape, dtype="uint8")
        debug_plate_color_index = 0
        debug_plate_pointer = 0

    for letter in letters:

        letter_mask = letter['letter_mask']

        recognized_letter, error_ratio = recognize_letter(letter_mask)

        licence_plate_letters.append({
            'recognized_letter': recognized_letter,
            'error_ratio': error_ratio,
            'position': letter['position'],
            'width': letter_mask.shape[1]
        })

        if DEBUG:
            debug_colored_letter = np.zeros((letter_mask.shape[0], letter_mask.shape[1], 3), dtype="uint8")
            debug_colored_letter[np.where(letter_mask > 0)] = debug_plate_letters_image_colors[debug_plate_color_index]

            debug_plate_color_index = (debug_plate_color_index + 1) % len(debug_plate_letters_image_colors)

            debug_plate_letters_image[
            0:debug_colored_letter.shape[0],
            debug_plate_pointer:debug_plate_pointer + letter_mask.shape[1]
            ] = debug_colored_letter

            debug_plate_pointer += letter_mask.shape[1]

    if DEBUG:
        cv2.imshow("debug plate", debug_plate_letters_image)

    if len(licence_plate_letters) > max_expected_letters:

        min_error_ratio_letters = sorted(licence_plate_letters, key=lambda letter: letter['error_ratio'])[
                                  :max_expected_letters]
        min_error_ratio_letters = sorted(min_error_ratio_letters, key=lambda letter: letter['position'])

    else:

        min_error_ratio_letters = sorted(licence_plate_letters, key=lambda letter: letter['position'])

    licence_plate = [letter['recognized_letter'] for letter in min_error_ratio_letters]

    # Dashes

    if plate_color == 'yellow' and len(licence_plate) == 6:

        gap_0_1 = min_error_ratio_letters[1]['position'] - min_error_ratio_letters[0]['position'] - \
                  min_error_ratio_letters[0]['width']

        gap_1_2 = min_error_ratio_letters[2]['position'] - min_error_ratio_letters[1]['position'] - \
                  min_error_ratio_letters[1]['width']

        if gap_0_1 > gap_1_2:

            licence_plate.insert(1, '-')
            licence_plate.insert(5, '-')

        else:
            licence_plate.insert(2, '-')

            gap_3_4 = min_error_ratio_letters[4]['position'] - min_error_ratio_letters[3]['position'] - \
                      min_error_ratio_letters[3]['width']
            gap_4_5 = min_error_ratio_letters[5]['position'] - min_error_ratio_letters[4]['position'] - \
                      min_error_ratio_letters[4]['width']

            if gap_3_4 > gap_4_5:
                licence_plate.insert(5, '-')
            else:
                licence_plate.insert(6, '-')

    return ''.join(licence_plate)


def recognize_letter(letter_mask):
    letter_mask_height = letter_mask.shape[0]

    # Generating template matching size if not already existing
    resize_templates(letter_templates, letter_mask_height)

    letter_mask_width = letter_mask.shape[1]

    min_mismatches = letter_mask.size
    min_mismatches_letter = "!"

    error_ratio = 1

    for letter_template in letter_templates:

        letter_template_matching_height = letter_template[letter_mask_height]

        letter_template_image = letter_template_matching_height["image"]

        letter_template_width = letter_template_image.shape[1]

        if letter_mask_width > letter_template_width:

            mismatches = count_mismatches_different_size(
                min_width_image=letter_template_image,
                max_width_image=letter_mask
            )
            comparison_size = letter_mask.size

        elif letter_mask_width < letter_template_width:

            mismatches = count_mismatches_different_size(
                min_width_image=letter_mask,
                max_width_image=letter_template_image
            )

            comparison_size = letter_template_image.size

        else:

            mismatches = count_mismatches_same_size(letter_template_image, letter_mask)
            comparison_size = letter_template_image.size

        if mismatches < min_mismatches:
            min_mismatches = mismatches
            min_mismatches_letter = letter_template["letter"]
            error_ratio = min_mismatches / comparison_size

    if min_mismatches_letter == "8":
        min_mismatches_letter = verify_8_letter(letter_mask)

    if min_mismatches_letter == "Z":
        min_mismatches_letter = verify_z_letter(letter_mask)

    return min_mismatches_letter, error_ratio


verify_8_critical_region_threshold = 0.6


def verify_8_letter(letter_mask):
    letter_mask_height = letter_mask.shape[0]

    critical_region_height = int(letter_mask_height / 6)
    critical_region_y = int((letter_mask_height / 2) - (critical_region_height / 2))

    critical_region_width = int(letter_mask.shape[1] / 3)

    critical_region = letter_mask[critical_region_y:critical_region_y + critical_region_height, 0:critical_region_width]

    critical_region_size = critical_region_width * critical_region_height

    if cv2.countNonZero(critical_region) / critical_region_size > verify_8_critical_region_threshold:
        return "B"
    else:
        return "8"


verify_z_critical_region_threshold = 0.38


def verify_z_letter(letter_mask):
    letter_mask_height = letter_mask.shape[0]

    critical_region_height = int(letter_mask_height / 6)

    critical_region = letter_mask[0:critical_region_height, ]

    critical_region_size = letter_mask.shape[0] * critical_region_height

    if cv2.countNonZero(critical_region) / critical_region_size > verify_z_critical_region_threshold:
        return "Z"
    else:
        return "2"
