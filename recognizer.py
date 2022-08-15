import cv2
import os
import shutil
import time

import tkinter
root = tkinter.Tk()
root.withdraw()

from tkinter import messagebox
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

class Verification():

  def __init__(self):
    self.cap = cv2.VideoCapture(0)

    self.cap.set(3,640) # width
    self.cap.set(4,480) # height

    if os.getenv("RESET_RECOG", None):
      if os.path.isdir("dataset"):
        shutil.rmtree("dataset")

    if os.path.isdir("dataset") == False:
      os.makedirs("dataset")
      self.gather_dataset()
    elif len([file for file in os.listdir("dataset") if file.endswith(".jpg")]) == 0:
      shutil.rmtree("dataset")
      os.makedirs("dataset")
      self.gather_dataset()
    
    self.dataset = [f"dataset/{file}" for file in os.listdir("dataset")]
    self.model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
      
  
  def gather_dataset(self):
    if os.getenv("IMAGE_COUNT", None):
      image_count = os.getenv("IMAGE_COUNT")
    else:
      image_count = 5
    
    if self.cap.isOpened():
      for i in range(int(image_count)):
        ret, frame = self.cap.read()
        if ret:
          self.extract_face(f"dataset/img_{i}.jpg", frame)
    
      self.cap.release()


  def extract_face(self, img_path, frame=None, required_size=(224, 224)):
    if frame.any() == None:
      pixels = pyplot.imread(img_path)
    else:
      pixels = frame

    #detect face
    detector = MTCNN()
    results = detector.detect_faces(pixels)

    if len(results) > 0:

      #resize&prepare face
      x1, y1, width, height = results[0]['box']
      x2, y2 = x1 + width, y1 + height
      face = pixels[y1:y2, x1:x2]

      #save cropped face
      cv2.imwrite(img_path, face)

    else:
      print("NO FACE WAS DETECTED!")
      while True:
        cv2.namedWindow("image")
        cv2.imshow("image",frame)
      #lock screen?

  def return_facearray(self, img_path, required_size=(224, 224)):
    image = cv2.imread(img_path)
    image = Image.fromarray(image)
    image = image.resize(required_size)
    
    return asarray(image)

  def get_embeddings(self, img_paths):
    faces = [self.return_facearray(img_path) for img_path in img_paths]

    samples = asarray(faces, 'float32')
    samples = preprocess_input(samples, version=2)

    yhat = self.model.predict(samples)

    return yhat

  def is_match(self, known_embedding, candidate_embedding, thresh=0.3):
    is_match = False
    score = cosine(known_embedding, candidate_embedding)
    
    if score <= thresh:
      is_match = True  
    
    return is_match

  def accept_login(self): 
    accept_login = False

    chance = 0

    if self.cap.isOpened():
      while chance != 5 and accept_login == False:
        ret, frame = self.cap.read()
        if ret:
          self.extract_face("frame.jpg", frame)
        
        img_paths = ["frame.jpg"] + self.dataset

        embeddings = self.get_embeddings(img_paths)
        if len(img_paths) == 2:
          accept_login = self.is_match(embeddings[0], embeddings[1])
        elif len(img_paths) > 2:
          for i in range(len(img_paths[1:])):
            accept_login = self.is_match(embeddings[0], embeddings[i])
        
        chance += 1
        
        #cleanup
        os.remove("frame.jpg")

      self.cap.release()

    return accept_login

def alert_user():
  pass

if __name__ == "__main__":
  verf = Verification()
  if os.getenv("TIME", None) == False:
    time = int(os.getenv("TIME")) #mins
  else:
    time = 5
  
  while True:
    time.sleep(time*60)
    if not verf.accept_login():
      messagebox.showwarning("WARNING", "You are not my user !! Logging out...")
      time.sleep(2)
      os.system("shutdown -l")
    
