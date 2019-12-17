import cv2
import numpy as np

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

showGraphs = True
showSteps = True


def get_blank_percentage(imgYellowScene):
	return 1 - np.count_nonzero(imgYellowScene) / (imgYellowScene.shape[0] * imgYellowScene.shape[1])


def apply_morphing(imgYellowScene, percentage):
	kernel = np.ones((5, 5), np.uint8)
	imgOpening = cv2.morphologyEx(imgYellowScene, cv2.MORPH_OPEN, kernel)
	imgClosing = cv2.morphologyEx(imgOpening, cv2.MORPH_CLOSE, kernel)
	
	if  0.995 < percentage <= 0.99:
		imgYellowDilated = cv2.dilate(imgYellowScene, kernel, iterations=3)
		print('case 1')
	if 0.99 < percentage <= 0.98:
		imgYellowDilated = cv2.dilate(imgYellowScene, kernel, iterations=2)
		print('case 2')
	if 0.98 < percentage <= 0.97:
		imgMedianBlur = cv2.medianBlur(imgYellowScene, 5)
		imgYellowDilated = cv2.dilate(imgMedianBlur, kernel, iterations=3)
		print('case 3')
	if 0.97 < percentage <= 0.96:
		imgMedianBlur = cv2.medianBlur(imgYellowScene, 5)
		imgYellowDilated = cv2.dilate(imgMedianBlur, kernel, iterations=3)
		print('case 4')
	else:
		imgMedianBlur = cv2.medianBlur(imgYellowScene, 5)
		imgYellowDilated = cv2.dilate(imgMedianBlur, kernel, iterations=3)
		print('case 5')

	return imgYellowDilated


def get_license_plate(imgMorphing):
	# Find x_max, x_min, y_max, y_min and crop out the image
	return None


def plate_detection(imgOriginalScene):
	# Increase contrast

	imgBlurred = cv2.GaussianBlur(imgOriginalScene, (5, 5), 0)
	imgHSVScene = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2HSV)
	imgYellowScene = cv2.inRange(imgHSVScene, (20, 100, 100), (110, 255, 255))

	percentage = get_blank_percentage(imgYellowScene)

	imgMorphing = apply_morphing(imgYellowScene, percentage)
	
	if showSteps:
		cv2.imshow("After Morphing", imgMorphing)
	
	imgLicensePlate = get_license_plate(imgMorphing)

	return imgLicensePlate
