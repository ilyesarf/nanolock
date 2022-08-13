from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import cv2

class Recognizer():

  def __init__(self):
    pass
    
  #gather dataset (multiple images is optional)
  
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

    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    yhat = model.predict(samples)

    return yhat

  def accept_login(self, known_embedding, candidate_embedding, thresh=0.5):
    accept_login = False
    score = cosine(known_embedding, candidate_embedding)
    
    if score <= thresh:
      accept_login = True  
    
    return accept_login

  def recognizer(self):
    pass
  
