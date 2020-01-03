import cv2
import numpy as np
from scipy.spatial import distance as dist


def get_histogram_correlation_grayscale(curr_frame, last_frame):
    gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    gray_last = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)

    bins = 10

    hist_curr = cv2.calcHist([gray_curr],[0],None,[bins],[0,256])
    hist_last = cv2.calcHist([gray_last],[0],None,[bins],[0,256])
    return cv2.compareHist(hist_curr, hist_last, 0)


def get_histogram_correlation(curr_frame, last_frame):
    hsv_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
    hsv_last = cv2.cvtColor(last_frame, cv2.COLOR_BGR2HSV)

    hsv_curr_half = hsv_curr[hsv_curr.shape[0] // 2:, :]
    h_bins = 50
    s_bins = 60
    histSize = [h_bins, s_bins]
    # hue varies from 0 to 179, saturation from 0 to 255
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges  # concat lists
    # Use the 0-th and 1-st channels
    channels = [0, 1]
    hist_curr = cv2.calcHist([hsv_curr], channels, None, histSize, ranges, accumulate=False)
    cv2.normalize(hist_curr, hist_curr, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    hist_last = cv2.calcHist([hsv_last], channels, None, histSize, ranges, accumulate=False)
    cv2.normalize(hist_last, hist_last, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    coeff = cv2.compareHist(hist_curr, hist_last, 0)
    return coeff


def get_histogram_difference(curr_frame, last_frame):
    hsv_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
    hsv_last = cv2.cvtColor(last_frame, cv2.COLOR_BGR2HSV)

    hsv_curr_half = hsv_curr[hsv_curr.shape[0] // 2:, :]
    h_bins = 50
    s_bins = 60
    histSize = [h_bins, s_bins]
    # hue varies from 0 to 179, saturation from 0 to 255
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges  # concat lists
    # Use the 0-th and 1-st channels
    channels = [0, 1]
    hist_curr = cv2.calcHist([hsv_curr], channels, None, histSize, ranges, accumulate=False)
    hist_curr = cv2.normalize(hist_curr, hist_curr, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).flatten()

    hist_last = cv2.calcHist([hsv_last], channels, None, histSize, ranges, accumulate=False)
    hist_last = cv2.normalize(hist_last, hist_last, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).flatten()

    distance = dist.euclidean(hist_curr, hist_last)
    return distance


def ECR(frame, prev_frame, width, height, crop=True, dilate_rate = 5):
    safe_div = lambda x,y: 0 if y == 0 else x / y
    if crop:
        startY = int(height * 0.3)
        endY = int(height * 0.8)
        startX = int(width * 0.3)
        endX = int(width * 0.8)
        frame = frame[startY:endY, startX:endX]
        prev_frame = prev_frame[startY:endY, startX:endX]

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray_image, 0, 200)
    dilated = cv2.dilate(edge, np.ones((dilate_rate, dilate_rate)))
    inverted = (255 - dilated)
    gray_image2 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    edge2 = cv2.Canny(gray_image2, 0, 200)
    dilated2 = cv2.dilate(edge2, np.ones((dilate_rate, dilate_rate)))
    inverted2 = (255 - dilated2)
    log_and1 = (edge2 & inverted)
    log_and2 = (edge & inverted2)
    pixels_sum_new = np.sum(edge)
    pixels_sum_old = np.sum(edge2)
    out_pixels = np.sum(log_and1)
    in_pixels = np.sum(log_and2)
    return max(safe_div(float(in_pixels),float(pixels_sum_new)), safe_div(float(out_pixels),float(pixels_sum_old)))


def get_mean_frame(frame):
    return np.sum(frame) / float(frame.shape[0] * frame.shape[1] * frame.shape[2])