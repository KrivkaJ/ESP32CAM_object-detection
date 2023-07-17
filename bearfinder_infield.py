#tento program je navrzen pro hledani medveda kamerou na predni strane robota
from ultralytics import YOLO
import numpy as np
import cv2 
import cvzone

#vstupni hodnoty programu 
searched_object = 'pottedplant' #nazev objektu z coco.names co chci najit 
vid_width = 720#sirka vide
vid_height = 480#vyska videa 

#konfigurace detekce objektu 

#nacte nazvy veci co muze detekovat ze souboru do listu 
classesfile='coco.names'
classNames=[]
with open(classesfile,'rt') as f:
    classNames=f.read().rstrip('\n').split('\n')

#zvoleni pouziteho modelu 
model = YOLO('yolov8n.pt')#v praxi pouzijeme jen nano a small

#nastaveni parametru videa 
cap = cv2.VideoCapture(0)#nacte video z kamery do promene
cap.set(3,vid_width)#sirka okna s videem
cap.set(4,vid_height)#vyska okna s videem

object_id = classNames.index(searched_object)# zjisti class id objektu co hledam

#smycka cteni jednotlivych framu videa 
while True:
     sucess, img = cap.read()
     results = model(img, stream = True)
     for r in results:
        boxes = r.boxes
        for box in boxes:
            cv2.line(img,(960,0),(960,1080),(255,0,255),thickness=2 )#vykresli na video primku stredem videa 
            cls = int(box.cls[0])#zjisti classu objektu
            if object_id == cls:#pokud se claasa objektu shoduje s objektem co hledam stane se nasledujici
                #bounding boxes
                x1,y1,x2,y2 = box.xyxy [0] #x1 je pozice leveho horniho rohu objektu v ose x, x2 je velikost objektu v ose x v px 
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)#prevedeni hodnot na int pro lepsi praci s nima 
                print('X=',x1,'Y=',y1,'W=',x2,'H=',y2)#vypisuje velikost objektu a jeho polohu v px 
                cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)#nakresli box okolo detekovane veci 
                #model confidence
                conf = box.conf[0]#jistota modelu 
                conf = float(conf*100)
                rounded_conf = int(conf)#zaokrouhli jistotu modelu na dve desetina mista 
                print('confidence:',rounded_conf)
                #claasnames
                cls = int(box.cls[0])#ulozi classu daneho objektu do promenne 
                print(classNames[cls])#vypise klassu objektu
                #object center 
                center_x,center_y = x1+(x2/2),y1+(y2/2)#vypocet stredu objektu pro lepsi lokalizaci medveda 
                center_x,center_y = int(center_x-x1/2), int(center_y-y1/2)#prevede hodnoty na int aby se dali pouzit ve funkci ukazujici stred 
                print('center:',center_x,center_y)#vypise udaje 
                cv2.circle(img, (center_x,center_y),10, (255,0,255), thickness=-1)
                #box on bounding box s nazvem claasy a confidence modelu 
                cvzone.putTextRect(img, f'{classNames[cls]}{rounded_conf}',(max(0,x1), max(35,y1)))#vykresli nazev classy objektu spolecne s confidence do videa 
                cv2.imshow('footage',img)#zobrazi frame
     key=cv2.waitKey(1000)#delay takze to vyhodnocuje jen jeden frame za sekundu pro odlehceni 
     if key==ord('q'):#pokud se zmackne klavesa q while true se brejkne 
        break
