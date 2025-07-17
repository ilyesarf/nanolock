
import os
import cv2
import base64
import json
import numpy as np
from PIL import Image
from hashlib import md5
import torch
import io
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine

class NoFaceDetected(Exception):
  pass


class Verification:
    def __init__(self):
        print("InceptionResnetV1 model loaded successfully (PyTorch)")
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.mtcnn = MTCNN(image_size=160, margin=0, device=self.device)
        self.dataset_dir = "nanolock/dataset/"
        if not os.path.isdir(self.dataset_dir):
            os.makedirs(self.dataset_dir)
        print(f"Dataset directory set to {self.dataset_dir}")


    def decode_img(self, b64enc_img):
        print("Decoding image...")
        img_data = base64.b64decode(b64enc_img)
        print(img_data[:20])  # Print first 20 bytes for debugging
        img_pil = Image.open(io.BytesIO(img_data)).convert('RGB')
        img_arr = np.array(img_pil)
        print("img_arr shape:", img_arr.shape)
        return img_arr

    def add_face(self, user_hash, b64enc_img):
        self.user_hash = user_hash
        img_path = f"{self.dataset_dir}{self.user_hash}.jpg"
        print(f"Adding face for user {self.user_hash} at {img_path}")
        img_arr = self.decode_img(b64enc_img)
        self.extract_face(img_path, img_arr)
  
    def extract_face(self, img_path, img_arr):
        # Use facenet-pytorch MTCNN for detection and alignment
        img_pil = Image.fromarray(img_arr)
        face = self.mtcnn(img_pil)
        if face is not None:
            # Convert tensor to numpy, scale to [0,255], and save as uint8 image
            face_img = face.permute(1, 2, 0).mul(255).byte().cpu().numpy()
            print("Premuted...", face_img.shape)
            cv2.imwrite(img_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
            print(f"Face saved to {img_path}")
        else:
            raise NoFaceDetected

    def return_facearray(self, img_path, required_size=(160, 160)):
        # Load image and convert to tensor for model
        image = Image.open(img_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(required_size),
            transforms.ToTensor()
        ])
        return transform(image)
  
    def get_embeddings(self, img_paths):
        faces = [self.return_facearray(img_path) for img_path in img_paths]
        faces_stack = torch.stack(faces).to(self.device)
        with torch.no_grad():
            embeddings = self.model(faces_stack).cpu().numpy()
        return embeddings

    def is_match(self, known_embedding, candidate_embedding, thresh=0.8):
        # Cosine similarity, lower is more similar
        score = cosine(known_embedding, candidate_embedding)
        return score <= thresh

    def accept_login(self, user_hash, b64enc_img):
        img_arr = self.decode_img(b64enc_img)
        self.extract_face("frame.jpg", img_arr)
        img_paths = ["frame.jpg", f"{self.dataset_dir}{user_hash}.jpg"]
        embeddings = self.get_embeddings(img_paths)
        accept_login = self.is_match(embeddings[0], embeddings[1])
        os.remove("frame.jpg")
        return accept_login
