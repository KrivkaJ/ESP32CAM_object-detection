import cv2 
import numpy as np

p_width = 140 #sirka hriste 
p_height = 280 #delka hriste 
margin_x = 25 # odsazeni od okraje okna v x 
margin_y = 25 # odsazeni od okraje okna v y 
playfield = np.zeros(((p_height*2)+50,(p_width*2)+50,3), dtype='uint8')
cv2.rectangle(playfield, (margin_x,margin_y),((p_width*2)+margin_x,(p_height*2)+margin_y),(255,255,255),thickness=2)
cv2.line(playfield,(25,305),(205,305),(255,255,255),thickness=2 )
cv2.line(playfield, (205,305),(205,305+140), (255,255,255),thickness=2)
cv2.line(playfield, (105,(p_height*2)+25),(105,305+140), (255,255,255),thickness=2)
#pos_x, pos_y = center_x/5,center_y/5
#cv2.circle(playfield, (pos_x,pos_y),10, (255,0,255), thickness=-1)
cv2.imshow('hriste', playfield)
cv2.waitKey(0)
