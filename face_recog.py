#IMPORT
import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet

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

cap = cv.VideoCapture(0)  # Try changing 1 to 0

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Confidence threshold
confidence_threshold = 0.5

# WHILE LOOP
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

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
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 10)
        cv.putText(frame, f"{final_name} ({max_confidence:.2f})", (x, y-10), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0, 0, 255), 3, cv.LINE_AA)

    cv.imshow("Face Recognition:", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()