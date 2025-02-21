import numpy as np
import cv2
import math
import time
import Validator

aspect_ratio_threshold = 3

letter_lower_limit = 0
letter_upper_limit = 110

centroid_y_threshold = .2
letter_component_max_width_ratio = 1 / 4
letter_component_height_min_ratio = 1 / 2

expected_number_of_letters = 5

kernel_2x2 = np.ones((2, 2), dtype='uint8')
kernel_3x3 = np.ones((2, 2), dtype='uint8')

letter_open_kernel = kernel_2x2

expected_letters = 6

DEBUG = False
debug_plate_letters_image_colors = np.array([
    [255, 255, 0],
    [0, 255, 255],
    [255, 0, 255]
], dtype='uint8')

histogram_bins_width = 5
histogram_bins = np.arange(0, 255, histogram_bins_width)

letter_threshold_ratio = 0.5

min_cc_area_ratio = 0.15

small_plate_threshold = 3000


def get_mask_centroid_x(mask):

    non_zero_points = np.where(mask > 0)[1]
    centroid_x = round(np.mean(non_zero_points))

    if math.isnan(centroid_x):
        return int(round(mask.shape[1] / 2))

    return int(centroid_x)


def load_letter_templates():
    raw_templates = [
        [cv2.imread('SameSizeLetters/1.bmp'), 'B'],
        [cv2.imread('SameSizeLetters/2.bmp'), 'D'],
        [cv2.imread('SameSizeLetters/3.bmp'), 'F'],
        [cv2.imread('SameSizeLetters/4.bmp'), 'G'],
        [cv2.imread('SameSizeLetters/5.bmp'), 'H'],
        [cv2.imread('SameSizeLetters/6.bmp'), 'J'],
        [cv2.imread('SameSizeLetters/7.bmp'), 'K'],
        [cv2.imread('SameSizeLetters/8.bmp'), 'L'],
        [cv2.imread('SameSizeLetters/9.bmp'), 'M'],
        [cv2.imread('SameSizeLetters/10.bmp'), 'N'],
        [cv2.imread('SameSizeLetters/11.bmp'), 'P'],
        [cv2.imread('SameSizeLetters/12.bmp'), 'R'],
        [cv2.imread('SameSizeLetters/13.bmp'), 'S'],
        [cv2.imread('SameSizeLetters/14.bmp'), 'T'],
        [cv2.imread('SameSizeLetters/15.bmp'), 'V'],
        [cv2.imread('SameSizeLetters/16.bmp'), 'X'],
        [cv2.imread('SameSizeLetters/17.bmp'), 'Z'],

        [cv2.imread('SameSizeNumbers/0.bmp'), '0'],
        [cv2.imread('SameSizeNumbers/1.bmp'), '1'],
        [cv2.imread('SameSizeNumbers/2.bmp'), '2'],
        [cv2.imread('SameSizeNumbers/3.bmp'), '3'],
        [cv2.imread('SameSizeNumbers/4.bmp'), '4'],
        [cv2.imread('SameSizeNumbers/5.bmp'), '5'],
        [cv2.imread('SameSizeNumbers/5temp.bmp'), '5'],
        [cv2.imread('SameSizeNumbers/6.bmp'), '6'],
        [cv2.imread('SameSizeNumbers/7.bmp'), '7'],
        [cv2.imread('SameSizeNumbers/8.bmp'), '8'],
        [cv2.imread('SameSizeNumbers/9.bmp'), '9'],
    ]

    result = []

    for letter_template_image, letter in raw_templates:

        letter_template_image = cv2.cvtColor(letter_template_image, cv2.COLOR_BGR2GRAY)

        height, width = letter_template_image.shape

        result.append({
            height: {
                'image': letter_template_image,
                'centroid_x': get_mask_centroid_x(letter_template_image),
                'width': width
            },
            'letter': letter,
            'default_height': height

        })

    return result


def resize_templates(templates, height):
    for template in templates:

        if height not in template:
            default_height = template['default_height']
            default_template_image = template[default_height]['image']

            default_width = template[default_height]['width']

            width = int(default_width * height / default_height)

            size = (width, height)
            resized_template_image = cv2.resize(default_template_image, dsize=size, interpolation=cv2.INTER_NEAREST)

            template.update({
                height: {
                    'image': resized_template_image,
                    'centroid_x': get_mask_centroid_x(resized_template_image)
                }
            })


letter_templates = load_letter_templates()


def count_mismatches(
        image_0,
        image_0_centroid,
        image_1,
        image_1_centroid,
        height
):

    # Left part

    if image_0_centroid > image_1_centroid:

        new_image_1 = np.zeros((height, image_1.shape[1] + image_0_centroid - image_1_centroid), dtype="uint8")
        new_image_1[:, image_0_centroid - image_1_centroid:] = image_1
        image_1 = new_image_1

    elif image_0_centroid < image_1_centroid:

        new_image_0 = np.zeros((height, image_0.shape[1] + image_1_centroid - image_0_centroid), dtype="uint8")
        new_image_0[:, image_1_centroid - image_0_centroid:] = image_0
        image_0 = new_image_0

    # Right part

    image_0_width = image_0.shape[1]
    image_1_width = image_1.shape[1]

    if image_0_width > image_1_width:

        new_image_1 = np.zeros((height, image_0_width), dtype="uint8")
        new_image_1[:, :image_1_width] = image_1
        image_1 = new_image_1

    elif image_1_width > image_0_width:

        new_image_0 = np.zeros((height, image_1_width), dtype="uint8")
        new_image_0[:, :image_0_width] = image_0
        image_0 = new_image_0

    # Total
    difference = cv2.absdiff(image_0, image_1)

    return cv2.countNonZero(difference)


def sharpen_image(plate):

    plate_gaussian = cv2.GaussianBlur(plate, (5, 5), 2.0)
    sharpened_plate = cv2.addWeighted(plate, 5, plate_gaussian, -4, 0, plate)

    return sharpened_plate


def get_small_plates_edges(plate, plate_color):

    plate = sharpen_image(plate)
    plate = cv2.resize(plate, dsize=(plate.shape[1] * 3, plate.shape[0] * 3))

    if DEBUG:
        cv2.imshow("sharpened image", plate)

    plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    letters_mask = cv2.inRange(plate_gray, 0, 90)

    edges = cv2.Canny(letters_mask, 100, 200)

    if DEBUG:
        cv2.imshow("letters mask", letters_mask)
        cv2.imshow("edges", edges)

    return plate, edges


small_plates_fixes = {
    'S': '5',
    '8': 'B',
    'B': '8'
}


def fix_small_plates(recognized_plate):

    def fix_mistake(plate_to_fix, from_letter, to_letter):

        if from_letter in plate_to_fix:

            s_positions = [pos for pos, char in enumerate(plate_to_fix) if char == from_letter]
            for s_position in s_positions:

                new_plate = plate_to_fix[:s_position] + to_letter + plate_to_fix[s_position + 1:]

                if Validator.pattern_check_dutch_license(new_plate):
                    return new_plate

        return None

    for fix in small_plates_fixes:

        fixed_plate = fix_mistake(recognized_plate, fix, small_plates_fixes[fix])

        if fixed_plate is not None:
            return fixed_plate

    return recognized_plate


def recognize_plate(plate, plate_color, force_sharpening=False):

    plate = fix_perspective(plate)

    is_small_plate = plate.size / 3 < small_plate_threshold

    # Detecting edges

    if force_sharpening:

        plate, edges = get_small_plates_edges(plate, plate_color)

    else:

        edges = cv2.Canny(plate, 120, 200)

        # Closing edges
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_2x2)

    if DEBUG:
        cv2.imshow("edges", edges)

    # Blob detection
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity=8)

    letters = []

    if num_labels < expected_letters + 1:

        if is_small_plate and not force_sharpening:
            return recognize_plate(plate, plate_color, True)
        else:
            return None

    height, width, _ = plate.shape

    min_component_height = int(letter_component_height_min_ratio * height)
    max_component_width = width * letter_component_max_width_ratio

    centroid_lower_y_limit = int(height / 2 - centroid_y_threshold * height)
    centroid_upper_y_limit = int(height / 2 + centroid_y_threshold * height)

    plate_grayscale = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

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

        letter_edges = edges[y:y + letter_height, x:x + letter_width]
        threshold = np.mean(letter_grayscale[np.where(letter_edges > 0)])

        letter_mask = np.zeros(letter_grayscale.shape, dtype='uint8')
        letter_mask[np.where(letter_grayscale < threshold)] = 255

        letter_mask = cv2.morphologyEx(letter_mask, cv2.MORPH_OPEN, letter_open_kernel)

        letters.append({
            'position': x,
            'letter_mask': letter_mask,
            'centroid_x': get_mask_centroid_x(letter_mask)
        })

    if len(letters) < expected_letters:
        if is_small_plate and not force_sharpening:
            return recognize_plate(plate, plate_color, True)
        else:
            return None

    licence_plate_letters = []

    if DEBUG:
        debug_plate_letters_image = np.zeros((plate.shape[0], plate.shape[1] + len(letters), 3), dtype='uint8')
        debug_plate_color_index = 0
        debug_plate_pointer = 0

    for letter in letters:

        letter_mask = letter['letter_mask']

        recognized_letter, error_ratio = recognize_letter(letter_mask, letter['centroid_x'])

        licence_plate_letters.append({
            'recognized_letter': recognized_letter,
            'error_ratio': error_ratio,
            'position': letter['position'],
            'width': letter_mask.shape[1],
            'centroid_x_abs': letter['centroid_x'] + letter['position']
        })

        if DEBUG:
            debug_colored_letter = np.zeros((letter_mask.shape[0], letter_mask.shape[1], 3), dtype='uint8')
            debug_colored_letter[np.where(letter_mask > 0)] = debug_plate_letters_image_colors[debug_plate_color_index]

            debug_plate_color_index = (debug_plate_color_index + 1) % len(debug_plate_letters_image_colors)

            debug_plate_letters_image[
                :debug_colored_letter.shape[0],
                debug_plate_pointer:debug_plate_pointer + letter_mask.shape[1]
            ] = debug_colored_letter

            debug_plate_pointer += letter_mask.shape[1]

            debug_plate_letters_image[:, debug_plate_pointer] = np.array([255, 255, 255], dtype='uint8')

            debug_plate_pointer += 1

    if DEBUG:

        cv2.imshow('debug plate', debug_plate_letters_image)

    if len(licence_plate_letters) > expected_letters:

        min_error_ratio_letters = sorted(licence_plate_letters, key=lambda letter: letter['error_ratio'])[
                                  :expected_letters]
        min_error_ratio_letters = sorted(min_error_ratio_letters, key=lambda letter: letter['position'])

    else:

        min_error_ratio_letters = sorted(licence_plate_letters, key=lambda letter: letter['position'])

    licence_plate = [letter['recognized_letter'] for letter in min_error_ratio_letters]

    # Dashes

    if plate_color == 'yellow' or plate_color == 'yellow_red_image' and len(licence_plate) == 6:

        gap_0_1 = min_error_ratio_letters[1]['centroid_x_abs'] - min_error_ratio_letters[0]['centroid_x_abs']
        gap_1_2 = min_error_ratio_letters[2]['centroid_x_abs'] - min_error_ratio_letters[1]['centroid_x_abs']

        if gap_0_1 > gap_1_2:

            licence_plate.insert(1, '-')
            licence_plate.insert(5, '-')

        else:
            licence_plate.insert(2, '-')

            gap_3_4 = min_error_ratio_letters[4]['centroid_x_abs'] - min_error_ratio_letters[3]['centroid_x_abs']
            gap_4_5 = min_error_ratio_letters[5]['centroid_x_abs'] - min_error_ratio_letters[4]['centroid_x_abs']

            if gap_3_4 > gap_4_5:
                licence_plate.insert(5, '-')
            else:
                licence_plate.insert(6, '-')

    recognized_plate = ''.join(licence_plate)

    if force_sharpening and not Validator.pattern_check_dutch_license(recognized_plate):

        # Plate was sharpened, but is not valid

        recognized_plate = fix_small_plates(recognized_plate)

    return recognized_plate


def recognize_letter(letter_mask, letter_mask_centroid_x):
    letter_mask_height = letter_mask.shape[0]

    # Generating template matching size if not already existing
    resize_templates(letter_templates, letter_mask_height)

    letter_mask_width = letter_mask.shape[1]

    min_mismatches = letter_mask.size
    min_mismatches_letter = '!'

    error_ratio = 1

    for letter_template in letter_templates:

        letter_template_matching_height = letter_template[letter_mask_height]

        letter_template_image = letter_template_matching_height['image']

        letter_template_width = letter_template_image.shape[1]

        if letter_mask_width > letter_template_width:

            comparison_size = letter_mask.size

        else:

            comparison_size = letter_template_image.size

        letter_template_centroid_x = letter_template_matching_height['centroid_x']

        mismatches = count_mismatches(
            image_0=letter_template_image,
            image_0_centroid=letter_template_centroid_x,
            image_1=letter_mask,
            image_1_centroid=letter_mask_centroid_x,
            height=letter_mask_height
        )

        if mismatches < min_mismatches:
            min_mismatches = mismatches
            min_mismatches_letter = letter_template['letter']
            error_ratio = min_mismatches / comparison_size

    return min_mismatches_letter, error_ratio


expected_aspect_ratio = 300 / 66
aspect_ratio_fix_threshold = 0.5


def fix_perspective(licence_plate):
    height, width, channels = licence_plate.shape

    aspect_ratio = width / height

    if np.abs(expected_aspect_ratio - aspect_ratio) > aspect_ratio_fix_threshold:

        # Aspect ratio is really off, must be fixed

        if aspect_ratio < expected_aspect_ratio:

            new_size = (int(expected_aspect_ratio * height), height)
        else:

            new_size = (width, int(width / expected_aspect_ratio))

        return cv2.resize(licence_plate, dsize=new_size)

    return licence_plate

