import re

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