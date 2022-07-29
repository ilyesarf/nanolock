import numpy as np
from PIL import Image
import cv2
import os
import shutil
import sys

class Recognizer():

  def __init__(self):
    if os.path.isdir("cascades") == False:
      print("cascades folder is not found")
      print("Leaving...")
      sys.exit()

    self.detector = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    self.recognizer = cv2.face.LBPHFaceRecognizer_create()

    self.cap = cv2.VideoCapture(0)

    self.cap.set(3,640) # set Width
    self.cap.set(4,480) # set Height
  
    if os.path.isdir("dataset") == False:
      os.makedirs("dataset")
      self.gather_dataset()
    elif len([name for name in os.listdir("dataset") if os.path.isfile(os.path.join("dataset", name))]) < 30:
      shutil.rmtree("dataset")
      os.makedirs("dataset")
      self.gather_dataset()

    if os.path.isdir("model") == False:
      os.makedirs("model")
      self.train()
    elif len(os.listdir("dataset/")) == 0:
      self.train()


  def gather_dataset(self):
    count = 0
    while count != 30:
        ret, frame = self.cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.detector.detectMultiScale(
          gray,
          scaleFactor = 1.3,
          minNeighbors=5,
          )

        for (x,y,w,h) in faces:
          cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
          count += 1

          cv2.imwrite(f"dataset/img.1.{count}.jpg", gray[y:y+h,x:x+w]) #1 is for the user id
        
          cv2.imshow('video', frame)
        
        k = cv2.waitKey(30) & 0xff
        if k == 27: # press 'ESC' to quit
            break

    self.cap.release()
    cv2.destroyAllWindows()
  
  def label_img(self):
    path = "dataset"
    imgpaths = [os.path.join(path,f) for f in os.listdir(path)]     
    samples=[]
    ids = []

    for imgpath in imgpaths:
        PIL_img = Image.open(imgpath).convert('L') #gray
        img_numpy = np.array(PIL_img,'uint8') #img to numpy

        id = int(os.path.split(imgpath)[-1].split(".")[1])
        faces = self.detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            samples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return samples,ids
  
  def train(self):
    dataset_path = "dataset"
   
    faces,ids = self.label_img()

    self.recognizer.train(faces, np.array(ids))
    self.recognizer.write('model/model.yml') #save model

  def recognize(self):
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    #init id
    id = 0

    names = ["Unknown", os.getlogin()]

    minW = 0.1*self.cap.get(3)
    minH = 0.1*self.cap.get(4)

    confidence = 0
    accept_login = False

    chance = 0

    while chance != 5 and accept_login == False:

      self.recognizer.read("model/model.yml") #load model

      ret, frame = self.cap.read()
      gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
      
      faces = self.detector.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
      )
      
      for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = self.recognizer.predict(gray[y:y+h,x:x+w])

        if round(100-confidence) >= 51:
          accept_login = True
        else:
          chance += 1

    self.cap.release()
    return accept_login

recognizer = Recognizer()
print(recognizer.recognize())
  
  
  
