#program na nalezani medveda pro webkameru pro jednodussi vyvoj
from ultralytics import YOLO
import numpy as np
import cv2 
import cvzone

#nastaveni modelu a nazvu veci 
classesfile='coco.names'
classNames=[]
with open(classesfile,'rt') as f:
    classNames=f.read().rstrip('\n').split('\n')
model = YOLO('yolov8n.pt')#zvoleni jaky model se pouzije 
#nastaveni parametru videa 
cap = cv2.VideoCapture(0)#nacte video z kamery do promene
cap.set(3,720)#sirka okna s videem
cap.set(4,480)#vyska okna s videem
#cteni a zobrazovani videa, jeden loop se rovna jeden frame 
while True:
     sucess, img = cap.read()
     results = model(img, stream = True)
     for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy [0] #x1 je pozice leveho horniho rohu objektu v ose x, x2 je velikost objektu v ose x v px 
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)#prevedeni hodnot na int pro lepsi praci s nima 
            print('X=',x1,'Y=',y1,'W=',x2,'H=',y2)#vypisuje velikost objektu a jeho polohu v px 
            center_x,center_y = x1+(x2/2),y1+(y2/2)#vypocet stredu objektu pro lepsi lokalizaci medveda 
            print(center_x,center_y)#vypise udaje 
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)#nakresli box okolo detekovane veci 
            conf = box.conf[0]#jistota modelu 
            conf = float(conf*100)
            rounded_conf = int(conf)#zaokrouhli jistotu modelu na dve desetina mista 
            print('confidence:',rounded_conf)
            #class names 
            cls = int(box.cls[0])#ulozi classu daneho objektu do promenne 
            cvzone.putTextRect(img, f'{classNames[cls]}{rounded_conf}',(max(0,x1), max(35,y1)))#vykresli nazev classy objektu spolecne s confidence do videa 
            print(classNames[cls])#vypise klassu objektu
     cv2.imshow('footage',img)
     key=cv2.waitKey(1000)#delay takze to vyhodnocuje jen jeden frame za sekundu pro odlehceni 
     if key==ord('q'):#pokud se zmackne klavesa q while true se brejkne 
        break