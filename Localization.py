import cv2
import numpy as np
import matplotlib.pyplot as plt

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

THRESHOLDING_ROW = 30
THRESHOLDING_COL = 100

showGraphs = False
showSteps = False

def plate_detection(imgOriginalScene):

	imgHSVScene = cv2.cvtColor(imgOriginalScene, cv2.COLOR_BGR2HSV)
	imgYellowScene = cv2.inRange(imgHSVScene, (20, 100, 100), (30, 255, 255))
	kernel = np.ones((5, 5), np.uint8)
	imgYellowDilated = cv2.dilate(imgYellowScene, kernel, iterations=3)

	if showSteps:
		cv2.imshow("Yellow", imgYellowScene)
		cv2.imshow("Dilated", imgYellowDilated)

	mask_row = np.zeros(imgYellowDilated.shape[0])
	mask_col = np.zeros(imgYellowDilated.shape[1])

	for row in imgYellowDilated:
		for i, pixel in enumerate(row):
			if pixel > 0:
				if i < 480:
					mask_row[i] += 1

	for j, row in enumerate(imgYellowDilated):
		for pixel in row:
			if pixel > 0:
				if j < imgYellowDilated.shape[1]:
					mask_col[j] += 1

	start_row = 0
	end_row = 0
	start_col = 0
	end_col = 0

	for i, count_in_row in enumerate(mask_row):
		if count_in_row > THRESHOLDING_ROW:
			if start_row == 0:
				start_row = i
			end_row = i

	for j, count_in_col in enumerate(mask_col):
		if count_in_col > THRESHOLDING_COL:
			if start_col == 0:
				start_col = j
			end_col = j

	if showGraphs:
		plt.plot(mask_row)
		plt.title('row')
		plt.show()

		plt.plot(mask_col)
		plt.title('col')
		plt.show()

	if start_col > 0 and start_row > 0:
		imgCroppedPlate = imgOriginalScene[start_col:end_col, start_row:end_row]
		return imgCroppedPlate
	else:
		print('nothing found')
		return None
