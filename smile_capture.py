# -*- coding: utf-8 -*-
"""
Created on Fri May 22 13:48:46 2020

@author: divyesh jolapara
"""
import cv2

from datetime import datetime


def detect(gray,frame):
    """
    DETECT TAKES IN GRAY SCALE FRAME AND COLORED FRAME
    IT DETECTS A FACE AT FIRST AND FROM THE FACE IT DETECTS
    A SMILE. IT RETURNS THE FRAME AND TRUE IF IT DETECTS A SMILE.
    
    """
    capture_pic=False
    
    # FACE DETECTION
    faces=face_cascade.detectMultiScale(gray,1.05,5)
    
    # DRAW A RECTANGLE OVER THE FACE
    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        
        # SMILE DETECTION
        smiles=smile_cascade.detectMultiScale(roi_gray,1.8,50)
        
        if len(smiles)>0:
             capture_pic=True
             
        # DRAW A RECTANGLE OVER THE SMILE
        for sx,sy,sw,sh in smiles:
           cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(255,0,0),2) 
        #print(smiles)
    return frame,capture_pic

face_cascade=cv2.CascadeClassifier("C:/Users/divye/Desktop/haarcascadeFiles/haar-cascade-files-master/haarcascade_frontalface_default.xml")
smile_cascade=cv2.CascadeClassifier("C:/Users/divye/Desktop/haarcascadeFiles/haar-cascade-files-master/haarcascade_smile1.xml")

video=cv2.VideoCapture(0)


while True:
    
    # SETTING IMAGE NAME AS SMILE_PIC_DATE_TIME.JPG
    now=datetime.now()
    dt_string = now.strftime("_%d%m%Y_%H%M%S")
    
    # READ CAMERA FRAME BY FRAME 
    check,frame=video.read()
    
    # CONVERT TO GRAY SCALE FOR LESS COMPUTATION
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    # CALLING FUNCTION DETECT
    canvas,take_pic=detect(gray,frame)
    
    # IF A SMILE IS DETECTED CAPTURE THE IMAGE AND SAVE IT
    if take_pic == True:
        
        cv2.imwrite("smile_pic"+dt_string+".jpg",frame)
        cv2.putText(canvas, '*****', (50,50), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (255,0,0), 2, cv2.LINE_AA) 
    key=cv2.waitKey(1)
    
    # SHOWS VIDEO  (FLIP IS USED TO FLIP THE IMAGE TO SHOW 
    # SCREEN AS A MIRROR)
    cv2.imshow("VIDEO...",cv2.flip(canvas,1,0))
    
    # q/Q TO QUIT TO STOP
    if key == ord('q') or key==ord('Q'):
        break
    
video.release()
cv2.destroyAllWindows()



















