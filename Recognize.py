import cv2
import numpy as np
import os

dictLetters = {0: "B", 1: "D", 2: "F", 3: "G", 4: "H", 5: "J", 6: "K", 7: "L", 8: "M", 9: "N", 10: "P", 11: "R", 12: "S", 13: "T", 14: "V", 15: "X", 16: "Z"}
dictNumbers = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "5", 7: "6", 8: "7", 9: "8", 10: "9"}

"""
In this file, you will define your own segment_and_recognize function.
To do:
	1. Segment the plates character by character
	2. Compute the distances between character images and reference character images(in the folder of 'SameSizeLetters' and 'SameSizeNumbers')
	3. Recognize the character by comparing the distances
Inputs:(One)
	1. plate_imgs: cropped plate images by Localization.plate_detection function
	type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
Outputs:(One)
	1. recognized_plates: recognized plate characters
	type: list, each element in recognized_plates is a list of string(Hints: the element may be None type)
Hints:
	You may need to define other functions.
"""
def segment_and_recognize(plate_imgs):
	cells = get_cells_from_plate(plate_imgs)
	license_plate = ""
	for cell in cells:
		license_plate = license_plate + template_matching(cell)
	return license_plate

"""
	Given the image of a plate, break down the plate into cells each containing a potential character
"""
def get_cells_from_plate(plate_img):
	cells = []
	# The goal is to segment the plate image and crop out each of the pictures of the characters

	return cells

"""
	cell_img should be GRAYSCALE image already
"""
def get_matching(param, n, cell_img):
	results = []
	for i in range(n):
		file_path = param + str(i + 1) + ".bmp"
		template = cv2.imread(file_path)
		templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
		w, h = templateGray.shape[::-1]
		res = cv2.matchTemplate(cell_img, templateGray, cv2.TM_CCOEFF_NORMED)
		results.append(max(res))
	return results

"""
	Returns the character with the highest probability of matching with the samples in memory.
	Uses dictionaries to get the result from the index of the max of the function.
"""
def template_matching(cell_img):
	results_numbers = get_matching("/SameSizeNumbers/", 9, cell_img)
	results_letters = get_matching("/SameSizeLetters/", 17, cell_img)
	max_numbers = max(results_numbers)
	max_letters = max(results_letters)
	if max_letters > max_numbers:
		return dictLetters[results_letters.index(max_letters)]
	else:
		return dictNumbers[results_numbers.index(max_numbers)]
