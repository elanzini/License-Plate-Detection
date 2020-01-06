import numpy as np
import cv2

aspect_ratio_threshold = 3

letter_lower_limit = 0
letter_upper_limit = 110

centroid_y_threshold = .2
letter_component_max_width_ratio = 1 / 4
letter_component_height_min_ratio = 1 / 2

min_number_of_letters = 5

kernel_2x2 = np.ones((2, 2), dtype="uint8")
kernel_3x3 = np.ones((2, 2), dtype="uint8")

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


def recognize_plate(plate):

    height, width, _ = plate.shape

    # Filtering by aspect ratio

    aspect_ratio = width / height

    if aspect_ratio < aspect_ratio_threshold:
        print("invalid aspect ratio")
        return None

    # Filtering letters color

    plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    plate_black_mask = cv2.inRange(plate_gray, letter_lower_limit, letter_upper_limit)

    # Applying 2x2 erosion to reduce noise

    plate_black_mask = cv2.erode(plate_black_mask, np.ones((2, 2), dtype="uint8"))

    # Applying 2x2 dilation
    plate_black_mask = cv2.dilate(plate_black_mask, np.ones((2, 2), dtype="uint8"))

    cv2.imshow("black mask eroded", plate_black_mask)

    # Detecting connected components

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(plate_black_mask)

    if num_labels < min_number_of_letters:
        print("too few labels")
        return None

    letters = 0

    centroid_lower_y_limit = int(height / 2 - centroid_y_threshold * height)
    centroid_upper_y_limit = int(height / 2 + centroid_y_threshold * height)

    min_component_height = int(letter_component_height_min_ratio * height)

    black_mask_filtered = cv2.cvtColor(plate_black_mask, cv2.COLOR_GRAY2BGR)

    for label in range(num_labels):

        # Filtering by centroid position
        if centroids[label][1] < centroid_lower_y_limit or centroids[label][1] > centroid_upper_y_limit:

            black_mask_filtered[np.where(labels == label)] = (255, 0, 0)
            continue

        # Filtering by component width
        if stats[label, cv2.CC_STAT_WIDTH] > width * letter_component_max_width_ratio:

            black_mask_filtered[np.where(labels == label)] = (0, 255, 0)
            continue

        # Filtering by component height
        if stats[label, cv2.CC_STAT_HEIGHT] < min_component_height:

            black_mask_filtered[np.where(labels == label)] = (0, 0, 255)
            continue

        letters += 1

    cv2.imshow("black mask filtered", black_mask_filtered)


histogram_bins_width = 5
histogram_bins = np.arange(0, 255, histogram_bins_width)

letter_index = 0


def recognize_plate_edges(plate):

    height, width, _ = plate.shape
    plate_grayscale = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    # Detecting edges

    edges = cv2.Canny(plate, 100, 200)

    # Applying dilation

    edges = cv2.dilate(edges, np.array((2,2), dtype="uint8"))

    cv2.imshow("edges", edges)

    # Blob detection
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges)

    centroid_lower_y_limit = int(height / 2 - centroid_y_threshold * height)
    centroid_upper_y_limit = int(height / 2 + centroid_y_threshold * height)

    min_component_height = int(letter_component_height_min_ratio * height)

    max_component_width = width * letter_component_max_width_ratio

    letter_masks = {}

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

        letter_grayscale = plate_grayscale[y:y+letter_height, x:x+letter_width]

        letter_grayscale_histogram = np.histogram(letter_grayscale, bins=histogram_bins)[0]

        avg = np.average(letter_grayscale)

        last_black_bin_index = int(np.floor(avg / histogram_bins_width))

        main_black_bin_index = np.argmax(letter_grayscale_histogram[:last_black_bin_index])
        main_yellow_bin_index = np.argmax(letter_grayscale_histogram[last_black_bin_index + 1:]) + last_black_bin_index

        threshold = int((main_yellow_bin_index + main_black_bin_index) / 2) * histogram_bins_width

        letter_mask = np.zeros(letter_grayscale.shape, dtype="uint8")
        letter_mask[np.where(letter_grayscale < threshold)] = 255

        # Eroding letter_mask
        # letter_mask = cv2.erode(letter_mask, kernel_2x2)

        letter_masks.update({
            x: letter_mask
        })

    if len(letter_masks) < min_number_of_letters:
        print("not enough letters")
        return

    licence_plate = []

    global letter_index

    for letter_mask_key in sorted(letter_masks):

        letter_mask = letter_masks[letter_mask_key]

        cv2.imwrite("letters/" + str(letter_index) + ".bmp", letter_mask)
        letter_index += 1
        licence_plate.append(recognize_letter(letter_mask))

    print(''.join(licence_plate))


def recognize_letter(letter_mask):

    letter_mask_height = letter_mask.shape[0]

    # Generating template matching size if not already existing
    resize_templates(letter_templates, letter_mask_height)

    letter_mask_width = letter_mask.shape[1]

    min_mismatches = letter_mask.size
    min_mismatches_letter = "!"

    for letter_template in letter_templates:

        letter_template_matching_height = letter_template[letter_mask_height]

        letter_template_image = letter_template_matching_height["image"]

        letter_template_width = letter_template_image.shape[1]

        if letter_mask_width > letter_template_width:

            mismatches = count_mismatches_different_size(
                min_width_image=letter_template_image,
                max_width_image=letter_mask
            )

        elif letter_mask_width < letter_template_width:

            mismatches = count_mismatches_different_size(
                min_width_image=letter_mask,
                max_width_image=letter_template_image
            )

        else:

            mismatches = count_mismatches_same_size(letter_template_image, letter_mask)

        if mismatches < min_mismatches:
            min_mismatches = mismatches
            min_mismatches_letter = letter_template["letter"]

    if min_mismatches_letter == "8":
        min_mismatches_letter = verify_8_letter(letter_mask)

    if min_mismatches_letter == "Z":
        min_mismatches_letter = verify_z_letter(letter_mask)

    return min_mismatches_letter


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

    print(cv2.countNonZero(critical_region) / critical_region_size)

    if cv2.countNonZero(critical_region) / critical_region_size > verify_z_critical_region_threshold:
        return "Z"
    else:
        return "(2)"

