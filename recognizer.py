import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if os.path.isdir("/etc/nanolock"):
  os.chdir("/etc/nanolock")
else:
  os.makedirs("/etc/nanolock")
  os.chdir("/etc/nanolock")

import cv2
import sys
import json
import time
import shutil
import random
import signal
import subprocess as sp, shlex
import webbrowser
import smtplib, ssl
from multiprocessing import Value
from multiprocessing import Process
from tkinter import messagebox
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from flask import Flask, request, render_template


class NoFaceDetected(Exception):
  pass

class Verification:

  def __init__(self):

    if os.getenv("RESET_VERF", None):
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
          try:
            self.extract_face(f"dataset/img_{i}.jpg", frame)
          except NoFaceDetected:
            messagebox.showwarning("WARNING", "Couldn't generate dataset!")
            sys.exit(0)

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
    is_match = -1
    score = cosine(known_embedding, candidate_embedding)
    
    if score <= thresh:
      is_match = 1  
    
    return is_match

  def accept_login(self): 
    cap = cv2.VideoCapture(0)

    cap.set(3,640) # width
    cap.set(4,480) # height

    accept_login = -1 #-1 is wrong, 0 is not detected, 1 is correct

    chance = 0

    while chance <= 6 and accept_login == -1:
        try:
          ret, frame = cap.read()
          if ret:
            self.extract_face("frame.jpg", frame)
          
          img_paths = ["frame.jpg"] + self.dataset

          embeddings = self.get_embeddings(img_paths)
          if len(img_paths) == 2:
            accept_login = self.is_match(embeddings[0], embeddings[1])
          elif len(img_paths) > 2:
            for i in range(len(img_paths[1:])):
              accept_login = self.is_match(embeddings[0], embeddings[i])
          

          if chance == 6:
            break

          chance += 1
          
          os.remove("frame.jpg")

        except NoFaceDetected:
          accept_login = 0

    cap.release()

    return accept_login

class Alert:

  def __init__(self):
    
    if os.path.exists("creds.json"):
      self.load_configuration()
    else:
      self.setup_configuration()
  
  #Configuration
  def setup_configuration(self):
    self.gmail_addr_recv = input("Insert Gmail address to receive alerts: ")
    print()
    self.gmail_addr_send = input("Insert Gmail address to send alerts: ")
    print()
    self.gmail_passwd = input("Insert Gmail adress app password: ")

    creds = {"gmail_addr_recv": self.gmail_addr_recv, "gmail_addr_send": self.gmail_addr_recv, "gmail_passwd": self.gmail_passwd}

    #save creds to json file
    json.dump(creds, open("creds.json", "w"))
  
  def load_configuration(self):
    #load creds from json file
    creds = json.load(open("creds.json", "r"))

    self.gmail_addr_recv = creds["gmail_addr_recv"]
    self.gmail_addr_send = creds["gmail_addr_send"]
    self.gmail_passwd = creds["gmail_passwd"]
  
  #Pop-ups
  def no_face_code_check(self, code):
    app = Flask(__name__)

    success = Value('i', 0)
    @app.route('/get_code', methods=["GET", "POST"])
    def get_code():
      if request.method == "POST":
        code_input = request.form["code"]

        if int(code_input) == code:
          success.value = 1
          os.kill(os.getpid(), signal.SIGINT) #stop server
        
      return render_template('index.html')

    server = Process(target=app.run)
    
    server.start()
    webbrowser.open("localhost:5000/get_code")
    time.sleep(10)

    server.terminate()
    server.join()
    
    if success.value == 0:
      time.sleep(1)
      messagebox.showwarning("WARNING", "Wrong Code!! Logging out...")
      time.sleep(2)
      logout_cmd = "/bin/bash -c logout"
      sp.run(shlex.split(logout_cmd))

  #Warning Emails
  def send_email(self, msg): #only works with gmail
    port = 587
    gmail_server = "smtp.gmail.com"
    context = ssl.create_default_context()
    
    with smtplib.SMTP(gmail_server, port) as server:
      server.ehlo()
      server.starttls(context=context)
      server.ehlo()
      server.login(self.gmail_addr_send, self.gmail_passwd)
      server.sendmail(self.gmail_addr_send, self.gmail_addr_recv, msg)

  def alert_no_face(self, code):
    msg = MIMEMultipart()
    msg["Subject"] = "NANOLOCK: no face was detected"
    msg["From"] = self.gmail_addr_send
    msg["To"] = self.gmail_addr_recv

    msg_body = f"Code to access: {code}"
    msg.attach(MIMEText(msg_body, "plain"))
    
    try:
      self.send_email(msg.as_string())
    except Exception:
      pass

    time.sleep(7)

    self.no_face_code_check(code)


  def get_ip(self):
    import urllib.request

    external_ip = urllib.request.urlopen('https://ident.me').read().decode('utf8')

    return external_ip

  def alert_wrong_face(self):
    msg = MIMEMultipart()

    msg["Subject"] = "NANOLOCK: wrong face was detected"
    msg["From"] = self.gmail_addr_send
    msg["To"] = self.gmail_addr_recv

    msg_text = MIMEText(f"This face was detected: (IP: {self.get_ip()})")
    msg.attach(msg_text)

    image = MIMEImage(open("frame.jpg", 'rb').read(), name=os.path.basename("frame.jpg"))
    msg.attach(image)

    try:
      self.send_email(msg)
    except Exception:
      pass

    messagebox.showwarning("WARNING", "You are not my user !! Logging out...")
    time.sleep(2)
    logout_cmd = "/bin/bash -c logout"
    sp.run(shlex.split(logout_cmd))


if __name__ == "__main__":
  verf = Verification()
  alert = Alert()
  if os.getenv("TIME", None) == False:
    t = int(os.getenv("TIME")) #mins
  else:
    t = 5
  
  while True:
    time.sleep(t*60)
    accept_login = verf.accept_login()

    if accept_login == -1:
      alert.alert_wrong_face()
    elif accept_login == 0:
      alert.alert_no_face(random.randint(0, 9999))
    
      

