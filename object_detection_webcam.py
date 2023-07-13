#program na nalezani medveda pro webkameru pro jednodussi vyvoj
from ultralytics import YOLO
import numpy as np
import cv2 

model = YOLO('yolov8n.pt')#zvoleni jaky model se pouzije 
cap = cv2.VideoCapture(0)#nacte video z kamery do promene
while True:
     sucess, img = cap.read()
     results = model(img, stream = True)
     for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy [0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)#prevedeni hodnot na int pro lepsi praci s nima 
            print(x1,y1,x2,y2)#vypisuje velikost objektu a jeho polohu v px 
            cv2.rectangle(im,(x1,y1),(x2,y2),(255,0,255),3)#nakresli box okolo detekovane veci 
            cv2.imshow('footage',img)
            key=cv2.waitKey(1)