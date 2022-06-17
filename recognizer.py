import numpy as np
import cv2
import os

detector = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

cap.set(3,640) # set Width
cap.set(4,480) # set Height

def gather_dataset():
  count = 0
  while count != 30:
      ret, frame = cap.read()
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

      faces = detector.detectMultiScale(
        gray,
        scaleFactor = 1.3,
        minNeighbors=5,
        )

      for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        count += 1
        print(count)
        cv2.imwrite(f"dataset/img_{count}.jpg", gray[y:y+h,x:x+w])
      
        cv2.imshow('video', frame)
      
      k = cv2.waitKey(30) & 0xff
      if k == 27: # press 'ESC' to quit
          break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  if os.path.isdir("dataset") == False:
    os.makedirs("dataset")
    
  gather_dataset()
