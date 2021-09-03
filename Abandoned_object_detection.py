import numpy as np
import cv2
from tracker import *

# Initialize Tracker
tracker = ObjectTracker()


# location of first frame
firstframe_path =r'Frame.png'

firstframe = cv2.imread(firstframe_path)
firstframe_gray = cv2.cvtColor(firstframe, cv2.COLOR_BGR2GRAY)
firstframe_blur = cv2.GaussianBlur(firstframe_gray,(21,21),0)

# location of video
file_path ='cut.mp4'
cap = cv2.VideoCapture(file_path)

frameno = 0

kernel = np.ones((10,10),np.uint8) #higher the kernel, eg (20,20), more will be eroded or dilated


while (cap.isOpened()):
    ret, frame = cap.read()
    
    
    if ret==0:
        break
    
    frameno = frameno + 1
    # cv2.putText(frame,'%s%.f'%('Frameno:',frameno), (400,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blur = cv2.GaussianBlur(frame_gray,(21,21),0)

    frame_diff = cv2.absdiff(firstframe, frame)
    # cv2.imshow("frame diff",frame_diff)

    #Canny Edge Detection
    edged = cv2.Canny(frame_diff,80,200) 
    cv2.imshow('CannyEdgeDet',edged)
    
    thresh = cv2.morphologyEx(edged,cv2.MORPH_CLOSE, kernel, iterations=2)

    cv2.imshow('Morph_Close', thresh)

    #Create a copy of the thresh to find contours    
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    detections=[]
    count = 0
    for c in cnts:
        contourArea = cv2.contourArea(c)
        
        if contourArea > 50 and contourArea < 10000:
            count +=1

            (x, y, w, h) = cv2.boundingRect(c)

            detections.append([x, y, w, h])

    box_ids, abandoned_objects = tracker.update(detections)
    # print(abandoned)
    # box_ids = tracker.update(detections)
    
    # Draw rectangle and id over all tracked objects

    # for box_id in box_ids:
    #     x1, y1, w1, h1, object_id, dist = box_id
    #     cv2.putText(frame, str(object_id), (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    #     cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
    
    # Draw rectangle and id over all abandoned objects
    for objects in abandoned_objects:
        _, x2, y2, w2, h2, _ = objects
        cv2.putText(frame, "Abandoned object detected", (x2, y2 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)

    cv2.imshow('main',frame)
    if cv2.waitKey(15) == ord('q'):
        break

cv2.destroyAllWindows()