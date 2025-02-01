import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize webcam and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, detectionCon=0.7)
classifier = Classifier('Model_digits/keras_model.h5', 'Model_digits/labels.txt')

# Define constants
offset = 20
imgSize = 300  # Desired image size for the model input
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  # Labels for digits 0-9

while True:
    success, img = cap.read()  # Read frame from webcam
    imgOutput = img.copy()  # Create a copy of the frame for output display
    hands, img = detector.findHands(img, draw=False)  # Detect hands (no drawing on hand)

    if hands:  # If hands are detected
        hand = hands[0]
        x, y, w, h = hand["bbox"]  # Get bounding box of the hand

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # Create a white image for processing

        # Ensure cropping indices are within bounds
        imgCrop = img[max(0, y-offset):min(y+h+offset, img.shape[0]),
                      max(0, x-offset):min(x+w+offset, img.shape[1])]

        # Ensure the cropped image is not empty before proceeding
        if imgCrop.size != 0:
            imgCropShape = imgCrop.shape
            aspectRatio = h / w  # Calculate aspect ratio

            try:
                if aspectRatio > 1:  # If the hand is taller than it is wide
                    k = imgSize / h  # Calculate scale factor based on height
                    wCal = int(k * w)  # Adjust width accordingly
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))  # Resize the image
                    wGap = (imgSize - wCal) // 2  # Calculate the left and right gaps for centering
                    imgWhite[:, wGap:wGap + wCal] = imgResize  # Place resized image in the white canvas
                else:  # If the hand is wider than it is tall
                    k = imgSize / w  # Calculate scale factor based on width
                    hCal = int(k * h)  # Adjust height accordingly
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))  # Resize the image
                    hGap = (imgSize - hCal) // 2  # Calculate the top and bottom gaps for centering
                    imgWhite[hGap:hGap + hCal, :] = imgResize  # Place resized image in the white canvas

                # Perform prediction
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

                # Draw the result on the output image
                cv2.rectangle(imgOutput, (x - offset, y - offset - 50), 
                              (x - offset + 90, y - offset), (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)

            except Exception as e:
                print(f"Error during prediction: {e}")

        # Display the cropped hand and processed white image
        cv2.imshow("Cropped Hand", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # Display the final result on the webcam feed
    cv2.imshow("Image", imgOutput)

    # Exit the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
