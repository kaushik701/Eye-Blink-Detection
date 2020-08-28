import numpy as np
import cv2
print(cv2.__version__)

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades#

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml#
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

first_read = True

#starting the video capture
cap = cv2.VideoCapture(0)
while(True):
    ret,img = cap.read()
    #covering the recorded image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    if len(faces) > 0:
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

            #roi_face is face which is input to eye classifier
            roi_face = gray[y:y+h,x:x+w]
            roi_face_clr = img[y:y+h,x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_face,1.3,5)

            #examining the length of eyes object for eyes
        if (len(eyes)>=2):
                #check if program is running for detection
                if (first_read):
                    cv2.putText(img,'Eye detected press s to begin',(70,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),2)
                else:
                    cv2.putText(img,'Eyes open!',(70,70),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)

        else:
                if (first_read):
                    cv2.putText(img,'No eyes detected',(70,70),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),2)
                else:
                    #this will print on console and restart the algorithm
                    print('Blink Detected-------------!!')
                    cv2.waitKey(3000)
                    first_read = True
    else:
        cv2.putText(img,'No Face detected',(100,100),cv2.FONT_HERSHEY_PLAIN,
                    3,(0,255,0),2)

    #controlling the algorithm with keys
    cv2.imshow('img',img)
    a = cv2.waitKey(1)
    if (a==ord('q')):
        break
    elif(a==ord('s') and first_read):
        #this will start the detection
        first_read = False

cap.release()
cv2.destroyAllWindows()