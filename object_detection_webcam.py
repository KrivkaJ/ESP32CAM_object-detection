#program na nalezani medveda pro webkameru pro jednodussi vyvoj
from ultralytics import YOLO
import numpy as np
import cv2 
import cvzone
#souradnice bodu hriste pro kalibraci robota
#podle nich se bude robot stavet aby nasel souradnice robota spravne 
corner_ax = 68
corner_bx = 1518
corner_cx = 992
corner_dx = 186
corner_ay = 710
corner_by = 525
corner_cy = 366
corner_dy = 414
###########################
p_width = 140 #sirka hriste 
p_height = 280 #delka hriste 
margin_x = 25 # odsazeni od okraje okna v x 
margin_y = 25 # odsazeni od okraje okna v y 
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
object_id = classNames.index('pottedplant')# zjisti class id objektu co hledam
playfield = np.zeros(((p_height*2)+50,(p_width*2)+50,3), dtype='uint8')
cv2.rectangle(playfield, (margin_x,margin_y),((p_width*2)+margin_x,(p_height*2)+margin_y),(255,255,255),thickness=2)
cv2.line(playfield,(25,305),(205,305),(255,255,255),thickness=2 )
cv2.line(playfield, (205,305),(205,305+140), (255,255,255),thickness=2)
cv2.line(playfield, (105,(p_height*2)+25),(105,305+140), (255,255,255),thickness=2)
while True:
     sucess, img = cap.read()
     results = model(img, stream = True)
     for r in results:
        boxes = r.boxes
        for box in boxes:
             cv2.circle(img, (corner_dx,corner_dy),(5), (0,0,255),thickness=-1)
             cv2.circle(img, (corner_ax,corner_ay),(5), (0,0,255),thickness=-1)
             cv2.circle(img, (corner_bx,corner_by),(5), (0,0,255),thickness=-1)
             cv2.circle(img, (corner_cx,corner_cy),(5), (0,0,255),thickness=-1)
             cv2.line(img, (corner_ax,corner_ay),(corner_bx, corner_by), (0,0,255),thickness=2)#hrana ab
             cv2.line(img, (corner_bx, corner_by),(corner_cx,corner_cy), (0,0,255),thickness=2)#hrana bc
             cv2.line(img, (corner_cx,corner_cy),(corner_dx,corner_dy), (0,0,255),thickness=2)#hrana cd
             cv2.line(img, (corner_dx,corner_dy),(corner_ax,corner_ay), (0,0,255),thickness=2)#hrana da
             cls = int(box.cls[0])#zjisti classu objektu 
             if object_id == cls:#pokud se classa objektu shoduje s objektem co hledam stane se nasledujici 
                x1,y1,x2,y2 = box.xyxy [0] #x1 je pozice leveho horniho rohu objektu v ose x, x2 je velikost objektu v ose x v px 
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)#prevedeni hodnot na int pro lepsi praci s nima 
                print('X=',x1,'Y=',y1,'W=',x2,'H=',y2)#vypisuje velikost objektu a jeho polohu v px 
                cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)#nakresli box okolo detekovane veci 
                conf = box.conf[0]#jistota modelu 
                conf = float(conf*100)
                rounded_conf = int(conf)#zaokrouhli jistotu modelu na dve desetina mista 
                print('confidence:',rounded_conf)
                #class names 
                center_x,center_y = x1+(x2/2),y1+(y2/2)#vypocet stredu objektu pro lepsi lokalizaci medveda 
                center_x,center_y = int(center_x-x1/2), int(center_y-y1/2)#prevede hodnoty na int aby se dali pouzit ve funkci ukazujici stred 
                print('center:',center_x,center_y)#vypise udaje 
                cv2.circle(img, (center_x,center_y),10, (255,0,255), thickness=-1)
                cls = int(box.cls[0])#ulozi classu daneho objektu do promenne 
                cvzone.putTextRect(img, f'{classNames[cls]}{rounded_conf}',(max(0,x1), max(35,y1)))#vykresli nazev classy objektu spolecne s confidence do videa 
                print(classNames[cls])#vypise klassu objektu
                pos_x, pos_y = int(center_x/7),int((center_y/40)**2.15-360)
                print(pos_x, pos_y)#pozice medveda na mape 
                cv2.circle(playfield, (pos_x,pos_y),10, (255,0,255), thickness=-1)
     cv2.imshow('hriste', playfield)
     cv2.imshow('footage',img)
     key=cv2.waitKey(1000)#delay takze to vyhodnocuje jen jeden frame za sekundu pro odlehceni 
     if key==ord('q'):#pokud se zmackne klavesa q while true se brejkne 
        break