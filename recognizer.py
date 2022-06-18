import numpy as np
from PIL import Image
import cv2
import os
import shutil
import json

class Recognizer():

  def __init__(self):
    self.detector = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    self.recognizer = cv2.face.LBPHFaceRecognizer_create()

    if os.path.isfile("names.json") == False:
      self.names = {"names": ["Unknown  "]}
    
      self.name = input("What's your name? ")

      self.names["names"].append(self.name)

      json.dump(self.names, open("names.json", "w"))

    else:
      self.names = json.load(open("names.json", "r"))

      self.name = input("What's your name? ")

      if self.name not in self.names["names"]:
        self.names["names"].append(self.name)
        json.dump(self.names, open("names.json", "w"))
          
    if os.path.isdir("dataset") == False:
      os.makedirs("dataset")
      self.gather_dataset()
    elif len([name for name in os.listdir("dataset") if os.path.isfile(os.path.join("dataset", name))]) < 30:
      shutil.rmtree("dataset")
      os.makedirs("dataset")
      self.gather_dataset() 
    elif ("img.%s" % self.names["names"].index(self.name)) not in os.listdir("dataset"):
      self.gather_dataset() 

    if os.path.isdir("model") == False:
      os.makedirs("model")
      self.train()
    elif len(os.listdir("model/")) == 0:
      self.train()
    elif len(os.listdir("dataset")) > 30 and len(os.listdir("dataset"))%30 == 0: #check if new faces are added to the dataset
      shutil.rmtree("model")
      os.makedirs("model")
      self.train()


  def gather_dataset(self):

    cap = cv2.VideoCapture(0)

    cap.set(3,640) # set Width
    cap.set(4,480) # set Height
    
    count = 0
    while count != 30:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.detector.detectMultiScale(
          gray,
          scaleFactor = 1.3,
          minNeighbors=5,
          )

        for (x,y,w,h) in faces:
          cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
          count += 1

          cv2.imwrite("dataset/img.%s.%s.jpg" % (self.names["names"].index(self.name), count), gray[y:y+h,x:x+w]) 
        
          cv2.imshow('video', frame)
        
        k = cv2.waitKey(30) & 0xff
        if k == 27: # press 'ESC' to quit
            break

    cap.release()
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

    cap = cv2.VideoCapture(0)

    cap.set(3,640) # set Width
    cap.set(4,480) # set Height
    
    #init id
    id = 0

    minW = 0.1*cap.get(3)
    minH = 0.1*cap.get(4)

    while True:
        self.recognizer.read("model/model.yml") #load model

        ret, frame = cap.read()
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

          if (confidence < 100):
            id = self.names["names"][id]
            confidence = "  {0}%".format(round(100 - confidence))
          else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
          cv2.putText(
            frame, 
            f"Hey {str(id)}", 
            (x+5,y-5), 
            font, 
            1, 
            (255,255,255), 
            2
          )

          cv2.putText(
            frame, 
            str(confidence), 
            (x+5,y+h-5), 
            font, 
            1, 
            (255,255,0), 
            1
          ) 
        
        cv2.imshow('camera',frame) 
        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
  recognizer = Recognizer()
  recognizer.recognize()
  
  
  
