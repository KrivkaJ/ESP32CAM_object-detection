#program na nalezeni medveda pomoci kamery 
from ultralytics import YOLO
import numpy as np
import cv2 
import cvzone
import urllib.request

url = 'http://192.168.137.92/cam-hi.jpg'#url adresa esp cam 

model = YOLO('yolov8n.pt')#zvoleni jaky model se pouzije 

cap = cv2.VideoCapture(url)#nacte video z kamery do promene 
model = YOLO('yolov8n.pt')
while True:
    # Read a frame from the video stream
    img_resp=urllib.request.urlopen(url)
    imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
    #ret, frame = cap.read()
    im = cv2.imdecode(imgnp,-1)
    results = model(im, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy [0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)#prevedeni hodnot na int pro lepsi praci s nima 
            print(x1,y1,x2,y2)#vypisuje velikost objektu a jeho polohu v px 
            cv2.rectangle(im,(x1,y1),(x2,y2),(255,0,255),3)#nakresli box okolo detekovane veci 
    cv2.imshow('footage',im)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()
