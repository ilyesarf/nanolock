import cv2
import os
import shutil
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

class Recognizer():

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
      image_count = 1
    
    if self.cap.isOpened():
      for i in range(int(image_count)):
        ret, frame = self.cap.read()
        if ret:
          cv2.imwrite(f"dataset/img_{i}.jpg", frame)
    
      self.cap.release()


  def extract_face(self, img, required_size=(224, 224)):
    pixels = pyplot.imread(img)

    #detect face
    detector = MTCNN()
    results = detector.detect_faces(pixels)

    #resize&prepare face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)

    return face_array

  def get_embeddings(self, imgs):
    faces = [self.extract_face(img) for img in imgs]

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
          cv2.imwrite("frame.jpg", frame)
        
        imgs = ["frame.jpg"] + self.dataset

        embeddings = self.get_embeddings(imgs)
        if len(imgs) == 2:
          accept_login = self.is_match(embeddings[0], embeddings[1])
        elif len(imgs) > 2:
          for i in range(len(imgs[1:])):
            accept_login = self.is_match(embeddings[0], embeddings[i])
        
        chance += 1
        
        #cleanup
        os.remove("frame.jpg")

      self.cap.release()

    return accept_login
  
r = Recognizer()

print(r.accept_login())