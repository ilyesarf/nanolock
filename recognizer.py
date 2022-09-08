
import cv2
import os
import shutil
import json
from numpy import asarray
from PIL import Image
from hashlib import md5
from keras_vggface.vggface import VGGFace
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot
from keras_vggface.utils import preprocess_input
from scipy.spatial.distance import cosine


class NoFaceDetected(Exception):
  pass

class Verification:

  def __init__(self):

    self.model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

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
    
    #add user
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
      
    elif len(os.listdir(self.dataset_path)) == 0 or len(os.listdir(self.dataset_path))%os.getenv("IMAGE_COUNT") != 0:
      shutil.rmtree(self.dataset_path)
      os.makedirs(self.dataset_path)
      self.gather_dataset() 

    elif self.dataset_path not in self.sub_datasets: #check if there's already data for user
      self.gather_dataset()
        
  def gather_dataset(self):
    cap = cv2.VideoCapture(0)

    cap.set(3,640) # width
    cap.set(4,480) # height
    if os.getenv("IMAGE_COUNT", None):
      image_count = os.getenv("IMAGE_COUNT")
    else:
      image_count = 5
    
    if cap.isOpened():
      for i in range(int(image_count)):
        ret, frame = cap.read()
        if ret:
          self.extract_face(f"nanolock/dataset/{md5(self.username.encode()).hexdigest()}/img.{i}.jpg", frame)

      cap.release()
  
  def extract_face(self, img_path, frame=None, required_size=(224, 224)):
    if not frame.any():
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
      print("Err")
      raise NoFaceDetected

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

  def accept_login(self, username): 
    dataset_path = f"nanolock/dataset/{md5(username.encode()).hexdigest()}"

    cap = cv2.VideoCapture(0)

    cap.set(3,640) # width
    cap.set(4,480) # height

    accept_login = False
    chance = 0

    while chance <= 5 and accept_login == False:
        ret, frame = cap.read()
        if ret:
          self.extract_face("frame.jpg", frame)
        
        img_paths = ["frame.jpg"] + [f"{dataset_path}/{img_path}" for img_path in os.listdir(dataset_path)]

        embeddings = self.get_embeddings(img_paths)
        if len(img_paths) == 2:
          accept_login = self.is_match(embeddings[0], embeddings[1])
        elif len(img_paths) > 2:
          for i in range(len(img_paths[1:])):
            accept_login = self.is_match(embeddings[0], embeddings[i]) 

        chance += 1
          
        os.remove("frame.jpg")


    cap.release()

    return accept_login
