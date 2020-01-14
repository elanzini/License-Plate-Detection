import os
import cv2
import time
import PlateRecognition
import Validator

# plate = '05-GH-JS'
# plate = '24-LSB-1'
# plate = '25-XV-LX'
# plate = '72-FP-RV'
plate = '88-XP-FS'
# plate = '89-NV-JP'
# plate = '98-JJ-GB'
# plate = 'TT-BJ-42'


plate_files = os.listdir('debug/plates/' + plate)

for plate_file in plate_files:

    plate_image = cv2.imread('debug/plates/' + plate + '/' + plate_file)

    cv2.imshow("plate", plate_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    recognized_plate = PlateRecognition.recognize_plate(plate_image, 'yellow')

    if recognized_plate is not None:
        print(recognized_plate)

    time.sleep(.1)

cv2.destroyAllWindows()
