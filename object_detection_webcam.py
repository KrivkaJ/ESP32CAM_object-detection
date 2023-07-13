#program na nalezani medveda pro webkameru pro jednodussi vyvoj
from ultralytics import YOLO
import numpy as np
import cv2 

model = YOLO('yolov8n.pt')#zvoleni jaky model se pouzije 
cap = cv2.VideoCapture(0)#nacte video z kamery do promene
cap.set(3,720)
cap.set(4,480)
while True:
     sucess, img = cap.read()
     results = model(img, stream = True)
     for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy [0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)#prevedeni hodnot na int pro lepsi praci s nima 
            print(x1,y1,x2,y2)#vypisuje velikost objektu a jeho polohu v px 
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)#nakresli box okolo detekovane veci 
            conf = box.conf[0]#jistota modelu 
            conf = float(conf*100)
            rounded_conf = int(conf)#zaokrouhli jistotu modelu na dve desetina mista 
            print('confidence:',rounded_conf)
     cv2.imshow('footage',img)
     cv2.waitKey(1000)#delay takze to vyhodnocuje jen jeden frame za sekundu pro odlehceni 