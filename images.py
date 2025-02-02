import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize webcam and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier('Model_digits/keras_model.h5', 'Model_digits/labels.txt')

offset=20
imgSize = 400
labels=['0','1','2','3','4','5','6','7','8','9']

while True:
    success, img = cap.read()  # Read frame from webcam
    imgOutput = img.copy()  # Create copy of frame
    hands, img = detector.findHands(img, draw=False)  # Disable drawing on hand

    if hands:
        hand = hands[0]
        
        x, y, w, h = hand["bbox"]  # Get bounding box

        imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255  # Create white image

        imgCrop = img[y-offset : y+h+offset , x-offset : x+w+offset]  # Crop hand region

        imgCropShape=imgCrop.shape

        aspectRatio=h/w

        if aspectRatio>1:
            k=imgSize/h
            wcal= (k*w)
            imgResize=cv2.resize(imgCrop,(int(wcal),imgSize))
            imgResizeShape=imgResize.shape
            wGap=math.ceil((imgSize-wcal)/2)
            imgWhite[:,wGap:wcal+wGap]=imgResize
            prediction,index=classifier.getPrediction(imgWhite,draw=False)

        else:
            k=imgSize/w
            hcal= (k*h)
            imgResize=cv2.resize(imgCrop,(imgSize,int(hcal)))
            imgResizeShape=imgResize.shape
            hGap=math.ceil((imgSize-hcal)/2)
            imgWhite[hGap:hcal+hGap , : ]=imgResize
        cv2.imshow("Cropped Hand", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
        cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                      (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset),
                      (x + w+offset, y + h+offset), (255, 0, 255), 4)

    cv2.imshow("Image", imgOutput)  # Show main video feed

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
