from genericpath import isdir
import numpy as np
from PIL import Image
import cv2
import os
import shutil
import sys

import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam


class Recognizer():

  def __init__(self):
    if os.path.isdir("cascades") == False:
      print("cascades folder is not found")
      print("Leaving...")
      sys.exit()
    
    if os.path.isdir("nanolock/dataset/") == False:
      os.makedirs("nanolock/dataset/")

    if os.getenv("SET_USER", None):
      self.username = input("Set username: ")
    else:
      self.username = os.getlogin()

    self.users = {"Unknown": 0}
    self.size = (244,244)

    self.detector = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    self.recognizer = cv2.face.LBPHFaceRecognizer_create()

    self.cap = cv2.VideoCapture(0)

    self.cap.set(3,640) # set Width
    self.cap.set(4,480) # set Height

    if os.getenv("RESET_RECOG", None):
      if os.path.isdir("dataset") == True: #and os.path.isdir("model") == True:
        shutil.rmtree("dataset")
        #shutil.rmtree("model")

    self.setup()
      
  def setup(self):
    self.dataset_path = f"dataset/{self.username}"

    #dataset checks
    if os.path.isdir(self.dataset_path) == False:
      os.makedirs(self.dataset_path)
      self.gather_dataset()

    elif len([name for name in os.listdir(self.dataset_path) if os.path.isfile(os.path.join(self.dataset_path, name))]) < 60:
      shutil.rmtree(self.dataset_path)
      os.makedirs(self.dataset_path)
      self.gather_dataset()

    self.augment_dataset()

  def gather_dataset(self):
    count = 0
    while count != 60:
        _, frame = self.cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.detector.detectMultiScale(
          gray,
          scaleFactor = 1.3,
          minNeighbors=5,
          )

        for (x,y,w,h) in faces:
          cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
          count += 1

          cv2.imwrite(f"dataset/{self.username}/{count}.jpg", gray[y:y+h,x:x+w])
        
          cv2.imshow('video', frame)
        
        k = cv2.waitKey(60) & 0xff
        if k == 27: # press 'ESC' to quit
            break

    self.cap.release()
    cv2.destroyAllWindows()
    
  def augment_dataset(self):

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
    './dataset',
    target_size=(224,224),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=True)

    #print(train_generator.class_indices.values())

    return train_datagen, train_generator

recognizer = Recognizer()



  
  
  
