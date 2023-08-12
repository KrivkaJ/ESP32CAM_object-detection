import cv2
import time
cap = cv2.VideoCapture(0)
something, img = cap.read()
cv2.imshow('nic',img)
time.sleep(1)



