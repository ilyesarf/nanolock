import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import shutil
import json
import numpy as np
from PIL import Image
from hashlib import md5
from keras_vggface.vggface import VGGFace
from mtcnn.mtcnn import MTCNN
#from matplotlib import pyplot
from keras_vggface.utils import preprocess_input
from scipy.spatial.distance import cosine
import base64

class NoFaceDetected(Exception):
  pass

class Verification:

  def __init__(self):

    self.model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

    self.dataset_dir = "nanolock/dataset/"
    if os.path.isdir(self.dataset_dir) == False:
      os.makedirs(self.dataset_dir)

  def decode_img(self, b64enc_img): #base64 img to np array
    import io

    img_data = base64.b64decode(b64enc_img)
    img_pil = Image.open(io.BytesIO(img_data))
    img_arr = np.array(img_pil)

    return img_arr

  def add_user(self, user_hash, b64enc_img):
    self.user_hash = user_hash
    img_path = f"{self.dataset_dir}{self.user_hash}.jpg"

    img_arr = self.decode_img(b64enc_img)

    self.extract_face(img_path, img_arr)

    #add user_hash to db
    if os.path.isfile("users.json") == False:
      self.users = {"user_hashes": []}

      self.users["user_hashes"].append(self.user_hash)

      json.dump(self.users, open("users.json", "w"))

    else:
      self.users = json.load(open("users.json", "r"))

      if self.user_hash not in self.users["user_hashes"]:
        self.users["user_hashes"].append(self.user_hash)
        json.dump(self.users, open("users.json", "w"))
  
  def extract_face(self, img_path, img_arr):

    #detect face
    detector = MTCNN()
    results = detector.detect_faces(img_arr)

    if len(results) > 0:

      #resize&prepare face
      x1, y1, width, height = results[0]['box']
      x2, y2 = x1 + width, y1 + height
      face = img_arr[y1:y2, x1:x2]

      #save cropped face
      cv2.imwrite(img_path, face)

    else:
      raise NoFaceDetected

  def return_facearray(self, img_path, required_size=(224, 224)):
    image = cv2.imread(img_path)
    image = Image.fromarray(image)
    image = image.resize(required_size)
    
    return np.asarray(image)
  
  def get_embeddings(self, img_paths):
    faces = [self.return_facearray(img_path) for img_path in img_paths]

    samples = np.asarray(faces, 'float32')
    samples = preprocess_input(samples, version=2)

    yhat = self.model.predict(samples)

    return yhat

  def is_match(self, known_embedding, candidate_embedding, thresh=0.35):
    is_match = False
    score = cosine(known_embedding, candidate_embedding)
    
    if score <= thresh:
      is_match = True
    
    return is_match

  def accept_login(self, user_hash, b64enc_img): 

    accept_login = False

    img_arr = self.decode_img(b64enc_img)
    self.extract_face("frame.jpg", img_arr)
      
    img_paths = ["frame.jpg", f"{self.dataset_dir}{user_hash}.jpg"]

    embeddings = self.get_embeddings(img_paths)
    accept_login = self.is_match(embeddings[0], embeddings[1])

    os.remove("frame.jpg")

    return accept_login
