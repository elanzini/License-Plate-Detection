import cv2
import numpy as np
import os

dictLetters = {0: "B", 1: "D", 2: "F", 3: "G", 4: "H", 5: "J", 6: "K", 7: "L", 8: "M", 9: "N", 10: "P", 11: "R", 12: "S", 13: "T", 14: "V", 15: "X", 16: "Z"}
dictNumbers = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "5"}

THRESHOLD_MSER = 1250
MAX_VAL = 100000000000

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
    Before resizing everything and therefore altering the initial shape
    it is better if you crop out EXACTLY the perfect box of the number and only THEN
    resize it to the same dimensions of the template you are trying to match.
    Since not all the digits are the same, look at the ratio between height and width.
    This way you will be able to discard some matches.
"""


def segment_and_recognize(plate_imgs):
    imgPlateClean = plate_morph(plate_imgs)
    cells = blob_detector(imgPlateClean)
    license_plate = ""
    for cell in cells:
        license_plate = license_plate + template_matching(cell)
    return license_plate


def plate_morph(img):
    # Grayscale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Noise reduction
    imgBlurred = cv2.medianBlur(imgGray, 1)
    # Normalization
    imgNormalized = np.zeros((img.shape[0], img.shape[1]))
    imgNormalized = cv2.normalize(imgBlurred,  imgNormalized, 0, 255, cv2.NORM_MINMAX)
    # Digitization
    ret, imgThresholding = cv2.threshold(imgNormalized ,100 ,255, cv2.THRESH_OTSU)
    # Invert colors
    imgInverse = cv2.bitwise_not(imgThresholding)
    # Resize to 125 * 600
    imgResized = cv2.resize(imgInverse, (600, 125))
    return imgResized


"""
    Given the image of a plate, break down the plate into cells each containing a potential character
"""
def get_cells_from_plate(img_plate):
    cells = []

    (h, w) = img_plate.shape[:2]
    image_size = h * w
    mser = cv2.MSER_create()
    mser.setMaxArea(image_size // 2)
    mser.setMinArea(10)

    regions, rects = mser.detectRegions(img_plate)
    # Sort the blobs by x-axis
    rects = sorted(rects, key=lambda rec: rec[0])

    # With the rects you can e.g. crop the letters
    for (x, y, w, h) in rects:
        if w * h > THRESHOLD_MSER:
            cells.append(img_plate[y:y + h, x:x + w])
            cv2.rectangle(img_plate, (x, y), (x + w, y + h), color=(255, 0, 255), thickness=1)

    return cells


'''
    Crops image given starting x,y, width and eight
'''
def crop(img,x, y, w, h):
    return img[y:y + h, x:x + w]


"""
    cell_img should be GRAYSCALE image already
"""
def get_matching(param, n, cell_img, letters=False):
    imgRatio, imgPostProcessed = preprocess_cell(cell_img)
    results = []
    i = 0
    if letters:
        i = 1
    while i < n:
        file_path = param + str(i) + ".bmp"
        template = cv2.imread(file_path)
        templateDilated = prepare_template(template)
        res = get_difference(imgPostProcessed, templateDilated)
        results.append(res)
        i = i + 1
    if imgRatio >= 2.50 and letters is False:
        results[1] = 0
    if imgRatio < 2.50 and letters is False:
        results[1] = MAX_VAL
    if letters is False:
        templateFiveTemp = cv2.imread(param + "5temp.bmp")
        templateFiveToCompare = prepare_template(templateFiveTemp)
        results.append(get_difference(imgPostProcessed, templateFiveToCompare))
    return results


'''
    Preprocessing of the template image to compare them to
'''
def prepare_template(template):
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    first_nonzero_col, last_nonzero_col = first_last_nonzero(templateGray, axis=0)
    templateToCompare = templateGray[0: 85, first_nonzero_col:min(55,last_nonzero_col)]
    templateDilated = cv2.dilate(templateToCompare, np.ones((3,3), np.uint8), iterations=1)
    return templateDilated


"""
    Returns the character with the highest probability of matching with the samples in memory.
    Uses dictionaries to get the result from the index of the max of the function.
"""
def template_matching(cell_img):
    results_numbers = get_matching("numbers/", 10, cell_img)
    results_letters = get_matching("letters/", 18, cell_img, True)
    min_numbers = min(results_numbers)
    min_letters = min(results_letters)
    if min_letters < min_numbers:
        return dictLetters[results_letters.index(min_letters)]
    else:
        return dictNumbers[results_numbers.index(min_numbers)]


'''
    Preprocesing the image: expecting BGR cropped image
    Steps:
    1. Grayscaling
    2. Blurring
    3. Normalization
    4. Threshold
    5. Inverse
    6. Cropping
    7. Resizing
'''
def preprocess_cell(img):
    # Crop out image
    first_nonzero_col, last_nonzero_col = first_last_nonzero(img, axis=0)
    first_nonzero_row, last_nonzero_row = first_last_nonzero(img, axis=1)
    imgCropped = img[first_nonzero_row: last_nonzero_row, first_nonzero_col:last_nonzero_col]
    # Ratio = width / height
    ratio = (last_nonzero_row - first_nonzero_row) /  (last_nonzero_col - first_nonzero_col)
    if ratio > 2.50:
        # Resize - 80 * 20 is the size of the template images with a one
        imgResized = cv2.resize(imgCropped, (20, 85))
    elif 2.00 < ratio < 2.50:
        # Resize - 80 * 40 is the size of the template images with a J
        imgResized = cv2.resize(imgCropped, (40, 85))
    elif 1.65 <= ratio < 1.85:
        # Resize
        imgResized = cv2.resize(imgCropped, (50, 85))
    elif 1.4 <= ratio < 1.65:
        # Resize
        imgResized = cv2.resize(imgCropped, (55, 85))
    elif 1.2 < ratio < 1.4:
        # Resize - 80 * 20 is the size of the template images with an M
        imgResized = cv2.resize(imgCropped, (65, 85))
    elif 1.0 <= ratio <= 1.2:
        # Resize - 80 * 20 is the size of the template images with an M
        imgResized = cv2.resize(imgCropped, (75, 85))
    else:
        # Resize - 85 * 55 is the size of the template images that are not a one or a default size letter
        imgResized = cv2.resize(imgCropped, (55, 85))
    return ratio, imgResized


'''
    Manual implementation of template matching.
    Returns the sum of the absolute value of the differences of the two images to compare
    The smaller the better.
'''
def get_difference(img, template):
    start = 0
    min_val = MAX_VAL
    while start + template.shape[1] <= img.shape[1]:
        SAD = np.sum(np.abs(img[0:0 + img.shape[0], start:start + template.shape[1]] - template))
        if SAD < min_val:
            min_val = SAD
        start = start + 1
    return min_val


'''
    Finds first and last occurrence of a nonzero element in a 2D array.
'''
def first_last_nonzero(arr, axis=0, invalid_val=-1):
    mask = arr!=0
    pos = np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)
    '''
    pos will look something like this, where each number represents an index
     array([ 0,  0,  1, -1])
    '''
    first_nonzero = -1
    last_nonzero = -1
    for i, element in enumerate(pos):
        if element >= 0:
            if first_nonzero < 0:
                first_nonzero = i
            else:
                last_nonzero = i
    return first_nonzero, last_nonzero


'''
    Manual implementation of a blob detector using BFS search
'''
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Blob:
    def __init__(self, min_x, max_x, min_y, max_y):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y


def get_neighbours(i, j, img):
    neighs = []
    if i > 0:
        if img[i-1][j] > 0 : neighs.append(Point(i-1,j))
        if j > 0:
            if img[i-1][j-1] > 0 : neighs.append(Point(i-1,j-1))
        if j < img.shape[1]-1:
            if img[i-1][j+1] > 0 : neighs.append(Point(i-1,j+1))
    if j > 0:
        if img[i][j-1] > 0 : neighs.append(Point(i,j-1))
        if i < img.shape[0]-1:
            if img[i+1][j-1] > 0 : neighs.append(Point(i+1,j-1))
    if j < img.shape[1]-1:
        if img[i][j+1] > 0 : neighs.append(Point(i,j+1))
        if i < img.shape[0]-1:
            if img[i+1][j+1] > 0 : neighs.append(Point(i+1,j+1))
    if i < img.shape[0]-1:
        if img[i+1][j] > 0 : neighs.append(Point(i+1,j))
    return neighs


def update_blob(point, blob):
    if point.x < blob.min_x:
        blob.min_x = point.x
    if point.x > blob.max_x:
        blob.max_x = point.x
    if point.y < blob.min_y:
        blob.min_y = point.y
    if point.y > blob.max_y:
        blob.max_y = point.y


def blob_detector(img):
    cells = []
    blobs = []
    visited = np.zeros((img.shape[0], img.shape[1]))
    stack = []
    for i, row in enumerate(img):
        for j, col in enumerate(row):
            if not visited[i][j] and img[i][j] > 0:
                visited[i][j] = True
                p = Point(i, j)
                # row (i) = x
                # col (j) = y
                b = Blob(i, i, j, j)
                stack.append(p)
                while len(stack) > 0:
                    point = stack.pop()
                    # check all neighbours
                    neighs = get_neighbours(point.x, point.y, img)
                    # update the blob
                    for neigh in neighs:
                        if not visited[neigh.x][neigh.y]:
                            visited[neigh.x][neigh.y] = True
                            update_blob(neigh, b)
                            # push the new points in the stack
                            stack.append(neigh)
                blobs.append(b)
            else:
                visited[i][j] = True
    # sort blobs by x
    blobs = sorted(blobs, key=lambda blob: blob.min_y)
    for blob in blobs:
        cells.append(crop_blob(img, blob))
    return cells


def crop_blob(img, blob):
    return img[blob.min_x: blob.max_x+1, blob.min_y:blob.max_y+1]


def get_ratio_blob(blob):
    return (blob.max_x - blob.min_x) / max((blob.max_y - blob.min_y),1)

#TODO verify first last zero is not necessary and eliminate it
#TODO why 5 is always B? Do some checking and try different morphological techinques