import cv2
import numpy as np
import face_recognition as fr
import os
from datetime import datetime

path='Students image'
images=[]
classname=[]
mylist=os.listdir(path)
#print(mylist)
for i in mylist:
    currentimage= cv2.imread(f'{path}/{i}')
    images.append(currentimage)
    classname.append(os.path.splitext(i)[0])
#print(classname)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

encodeListknown = findEncodings(images)
print("Encoding Done. Scaning.....")

cap=cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
 
    facesCurFrame = fr.face_locations(imgS)
    encodesCurFrame = fr.face_encodings(imgS,facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = fr.compare_faces(encodeListknown,encodeFace)
        faceDis = fr.face_distance(encodeListknown,encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if faceDis[matchIndex]< 0.50:
            name = classname[matchIndex].upper()
            markAttendance(name)
        else: name = 'Unknown'
            #print(name)
        y1,x2,y2,x1 = faceLoc
        y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
        cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            


    
    if cv2.waitKey(10)==ord('q'):
        break
    cv2.imshow('webcam',img)
   

    

