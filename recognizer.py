import numpy as np
from PIL import Image
import cv2
import os
import shutil
import json
from hashlib import md5

class Recognizer():

  def __init__(self):

    self.detector = cv2.CascadeClassifier('nanolock/cascades/haarcascade_frontalface_default.xml')
    self.recognizer = cv2.face.LBPHFaceRecognizer_create()

    if os.path.isdir("nanolock/dataset/") == False:
      os.makedirs("nanolock/dataset/")

  def get_subdirs(self, path): 
    directories = []

    for file in os.listdir(path):
      d = os.path.join(path, file)
      if os.path.isdir(d):
        directories.append(d)

    return directories 

  def hash_users(self, users):
    hashes = [md5(user.encode()).hexdigest() for user in users[1:]]

    return hashes

  def setup(self, username):
    self.username = username
    self.dataset_path = f"nanolock/dataset/{md5(self.username.encode()).hexdigest()}"

    self.sub_datasets = self.get_subdirs("nanolock/dataset") 

    if os.path.isfile("usernames.json") == False:
      self.usernames = {"usernames": ["Unknown"]}

      self.usernames["usernames"].append(self.username)

      json.dump(self.usernames, open("usernames.json", "w"))

    else:
      self.usernames = json.load(open("usernames.json", "r"))

      if self.username not in self.usernames["usernames"]:
        self.usernames["usernames"].append(self.username)
        json.dump(self.usernames, open("usernames.json", "w"))
    
    #dataset checks      
    if os.path.isdir(self.dataset_path) == False:
      os.makedirs(self.dataset_path)
      self.gather_dataset()

    elif len(os.listdir(self.dataset_path)) == 0 or len(os.listdir(self.dataset_path))%60 != 0:
      shutil.rmtree(self.dataset_path)
      os.makedirs(self.dataset_path)
      self.gather_dataset() 

    elif self.dataset_path not in self.sub_datasets: #check if there's already data for user
      self.gather_dataset()

    #model checks
    if os.path.isdir("nanolock/model") == False:
      os.makedirs("nanolock/model")
      self.train()

    elif len(os.listdir("nanolock/model/")) == 0:
      self.train()

    elif len(os.listdir(self.dataset_path)) > 60 and len(os.listdir(self.dataset_path))%60 == 0: #check if new faces are added to the dataset
      shutil.rmtree("nanolock/model")
      os.makedirs("nanolock/model")
      self.train()

  def gather_dataset(self):

    cap = cv2.VideoCapture(0)

    cap.set(3,640) # set Width
    cap.set(4,480) # set Height
    
    count = 0
    while count != 60:
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

        cv2.imwrite("nanolock/dataset/%s/img.%s.jpg" % (md5(self.username.encode()).hexdigest(), count), gray[y:y+h,x:x+w]) 

    cap.release()
  
  def label_img(self, usernames=None):

    datasets_path = "nanolock/dataset/"
    hashed_users = self.hash_users(self.usernames["usernames"])

    imgpaths = []
    for hashed_user in hashed_users:
      user_dataset = f"{datasets_path}{hashed_user}"
      for f in os.listdir(user_dataset):
        imgpaths.append(os.path.join(user_dataset,f))
  
    samples=[]
    ids = []

    for imgpath in imgpaths:
      PIL_img = Image.open(imgpath).convert('L') #gray
      img_numpy = np.array(PIL_img,'uint8') #img to numpy

      for hashed_user in hashed_users:

        if hashed_user == imgpath.split("/")[2]:
          for username in self.usernames["usernames"][1:]:
            if md5(self.username.encode()).hexdigest() == hashed_user:
              id = int(self.usernames["usernames"].index(username))

        else:
          print("Error: %s != %s" % (hashed_user, imgpath.split("/")[2]))

        faces = self.detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            samples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return samples,ids
  
  def train(self):   
    faces,ids = self.label_img()

    self.recognizer.train(faces, np.array(ids))
    self.recognizer.write("nanolock/model/model.yml") #save model

  def recognize(self, usernames):
    cap = cv2.VideoCapture(0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    
    #init id
    id = 0

    minW = 0.1*cap.get(3)
    minH = 0.1*cap.get(4)

    confidence = 0
    accept_login = False

    chance = 0

    while chance != 5 and accept_login == False:

      self.recognizer.read("nanolock/model/model.yml") #load model

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

        if round(100-confidence) >= 51:
          accept_login = True
        else:
          chance += 1

    cap.release()
    if accept_login:
      return usernames[id]

    return accept_login


#TODO:  
#Better facial recognition with PyTorch or TensorFlow
#Get video stream from JS (client web-browser) to Python (OpenCV)
