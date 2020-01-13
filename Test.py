import cv2
import LightLocalization
import LightPlateRecognition
import time

# filename = 'TrainingSet\Categorie IV\Video89_2.avi'
filename = 'trainingsvideo.avi'

cap = cv2.VideoCapture(filename)

prev_plates_count = 0

recognized_plates = {}

plates_count = 0

plate_colors = [
    'yellow',
    'yellow_red_image',
    'white'
]

while cap.isOpened():

    ret, frame = cap.read()

    if frame is None:
        break

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    for plate_color in plate_colors:

        plates_color, plate_images = LightLocalization.locate_plates(frame, plate_color)

        if len(plate_images) > 0:
            break

    for i in range(len(plate_images)):
        cv2.imshow("plate " + str(i), plate_images[i])

    if len(plate_images) < prev_plates_count:
        for i in range(len(plate_images), prev_plates_count + 1):
            cv2.destroyWindow("plate " + str(i))

    prev_plates_count = len(plate_images)

    for i in range(len(plate_images)):

        plate = LightPlateRecognition.recognize_plate(plate_images[i], plates_color)

        if plate is not None:
            plates_count += 1
            print("Plate " + str(i) + ": " + plate)

            if plate in recognized_plates:

                recognized_plates.update({
                    plate: recognized_plates[plate] + 1
                })

            else:

                recognized_plates.update({
                    plate: 1
                })

most_common_plate = "!"
most_common_plate_occurrences = 0

for plate in recognized_plates:
    if recognized_plates[plate] > most_common_plate_occurrences:
        most_common_plate_occurrences = recognized_plates[plate]
        most_common_plate = plate

second_most_common_plate = '!'
second_most_common_plate_occurrences = 0

for plate in recognized_plates:
    if recognized_plates[plate] > second_most_common_plate_occurrences and plate != most_common_plate:
        second_most_common_plate_occurrences = recognized_plates[plate]
        second_most_common_plate = plate

print(str(most_common_plate_occurrences) + " -> " + most_common_plate)
print(str(second_most_common_plate_occurrences) + " -> " + second_most_common_plate)
if plates_count == 0:
    print("Confidence: -1")
else:
    print("Confidence: " + str(most_common_plate_occurrences / plates_count))

cap.release()
cv2.destroyAllWindows()
