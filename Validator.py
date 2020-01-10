import re
import cv2

'''
    Method checks for valid patterns based on 
    https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_the_Netherlands
    
    The following patterns are supported and checked:
    [X = letter, 9 = number]
    XX-XX-99
    99-XXX-9
    9-XXX-99
    99-XX-XX
    XX-999-X
    X-999-XX
'''

dictLetters = {0: "B", 1: "D", 2: "F", 3: "G", 4: "H", 5: "J", 6: "K", 7: "L", 8: "M", 9: "N", 10: "P", 11: "R", 12: "S", 13: "T", 14: "V", 15: "X", 16: "Z"}
dictNumbers = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "5"}


def pattern_check_dutch_license(license_plate):
    if len(license_plate) != 8 or license_plate.count('-') != 2:
        return False

    if re.match("[A-Z][A-Z]\-[A-Z][A-Z]\-[0-9][0-9]", license_plate):
        return True

    if re.match("[0-9][0-9]\-[A-Z][A-Z][A-Z]\-[0-9]", license_plate):
        return True

    if re.match("[0-9]\-[A-Z][A-Z][A-Z]\-[0-9][0-9]", license_plate):
        return True

    if re.match("[0-9][0-9]\-[A-Z][A-Z]\-[A-Z][A-Z]", license_plate):
        return True

    if re.match("[A-Z][A-Z]\-[0-9][0-9][0-9]\-[A-Z]", license_plate):
        return True

    if re.match("[A-Z]\-[0-9][0-9][0-9]\-[A-Z][A-Z]", license_plate):
        return True

    else:
        return False


def verify_z_letter(letter_mask):
    letter_mask_height = letter_mask.shape[0]
    critical_region_height = int(letter_mask_height / 6)
    critical_region = letter_mask[0:critical_region_height, ]
    critical_region_size = letter_mask.shape[0] * critical_region_height
    if cv2.countNonZero(critical_region) / critical_region_size > 0.55:
        return "Z"
    else:
        return "2"


def verify_b_letter(letter_mask):
    letter_mask_height = letter_mask.shape[0]
    critical_region_height = int(letter_mask_height / 3 * 1.1)
    critical_region = letter_mask[0:critical_region_height, ]
    critical_region_size = letter_mask.shape[0] * critical_region_height
    if cv2.countNonZero(critical_region) / critical_region_size > 0.4:
        return "B"
    else:
        return "5"


def verify_j_letter(letter_mask):
    return letter_mask.shape[0]/letter_mask.shape[1] > 1.5


def evaluate(results_numbers, results_letters):
    min_numbers = min(results_numbers)
    min_letters = min(results_letters)

    if min_letters < min_numbers:
        return dictLetters[results_letters.index(min_letters)]
    else:
        return dictNumbers[results_numbers.index(min_numbers)]

