import cv2

cap = cv2.VideoCapture(0)
something, img = cap.read()
cv2.imshow('nic',img)
cv2.waitKey(0)



