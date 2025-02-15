# face recognition part II
#IMPORT
import cv2 as cv
import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
import requests
from io import BytesIO
from PIL import Image
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.colab import drive as gdrive

# Handle SSL certificate verification error
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#INITIALIZE
facenet = FaceNet()
faces_embeddings = np.load("/Users/salam9/Downloads/face_recognition/faces_embeddings_done_4classes.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
haarcascade = cv.CascadeClassifier("/Users/salam9/Downloads/face_recognition/haarcascade_frontalface_default.xml")
model = pickle.load(open("/Users/salam9/Downloads/face_recognition/svm_model_160x160.pkl", 'rb'))

# Confidence threshold
confidence_threshold = 0.5

# Authenticate and create the PyDrive client
auth.authenticate_user()
gauth = GoogleAuth()
gauth.CommandLineAuth()
drive = GoogleDrive(gauth)

# Mount Google Drive
gdrive.mount('/content/drive')

# Folder ID from the Google Drive link
folder_id = '1opnUrtRV1dbgPWOqPiTkHpXScMMbzTdl'

# List all files in the folder
file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()

# Process each file in the folder
for file in file_list:
    file_id = file['id']
    file_name = file['title']
    print(f"Processing file: {file_name}")
    file.GetContentFile(file_name)
    frame = cv.imread(file_name)
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
    for x, y, w, h in faces:
        img = rgb_img[y:y+h, x:x+w]
        img = cv.resize(img, (160, 160))  # 1x160x160x3
        img = np.expand_dims(img, axis=0)
        ypred = facenet.embeddings(img)
        face_name = model.predict(ypred)
        confidence = model.decision_function(ypred)
        max_confidence = np.max(confidence)
        final_name = "not_sure" if max_confidence < confidence_threshold else encoder.inverse_transform(face_name)[0]
        print(f"Detected: {final_name}, Confidence: {max_confidence}")
        # if final_name == "shoaib_alam":
        print(f"File ID: {file_id}")

print("Processing complete.")