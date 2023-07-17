#tento program je navrzen pro hledani medveda kamerou na predni strane robota
from ultralytics import YOLO
import numpy as np
import cv2 
import cvzone

#vstupni hodnoty programu 
searched_object = 'pottedplant' #nazev objektu z coco.names co chci najit 


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
cap.set(3,720)#sirka okna s videem
cap.set(4,480)#vyska okna s videem

object_id = classNames.index(searched_object)# zjisti class id objektu co hledam