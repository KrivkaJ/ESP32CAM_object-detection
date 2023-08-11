import cv2
import time

cap = cv2.VideoCapture(0)

while True:
    ahoj, img = cap.read()
    cv2.imshow('footage', img)
    cv2.waitKey(1)
