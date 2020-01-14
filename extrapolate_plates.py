import cv2
import Localization
import PlateRecognition
import time

filename = 'TrainingSet\Categorie III\Video61_2.avi'
# filename = 'trainingsvideo.avi'

cap = cv2.VideoCapture(filename)

prev_plates_count = 0

recognized_plates = {}

plates_count = 0

plate_colors = [
    'yellow',
    'yellow_red_image',
    'white'
]

frames_counter = 0

while cap.isOpened():

    ret, frame = cap.read()

    if frame is None:
        break

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    for plate_color in plate_colors:

        plates_color, plate_images = Localization.locate_plates(frame, plate_color)

        if len(plate_images) > 0:
            break

    for i, plate_image in enumerate(plate_images):

        cv2.imwrite('debug/plates/' + str(frames_counter) + "_" + str(i) + ".bmp", plate_image)

    frames_counter += 1

cap.release()
cv2.destroyAllWindows()
