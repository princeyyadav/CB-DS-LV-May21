import numpy as np
import cv2
import sys

##### LOAD TRAINING DATA #########
X_train = np.load("X_train_new.npy")
Y_train = np.load("Y_train.npy")
print(X_train.shape, Y_train.shape)

###### KNN MODEL ##################
sys.path.append("E:/CB-DS-LV-May21/DS/S22")
from mysklearn.NNeighbours import KNN

knn = KNN()

##### READ THE WEBCAM ##############
# 1. Get an image of shape (1,30k)

map = {0: 'angelina', 1: 'Princey'}
cap = cv2.VideoCapture(0) # object to interact with web cam

face_cascade = cv2.CascadeClassifier("../../S21-face-recognition/haarcascade_frontalface_alt.xml")

padding = 10
while True:

    ret, image = cap.read()
    if ret== False:
        continue

    # detect faces in the image
    faces = face_cascade.detectMultiScale(image)

    print(faces)
    for face in faces:
        x, y, w, h = face

        # slice the face out of the image
        face_section = image[ y-padding:y+h+padding, x-padding:x+w+padding ]
        face_section = cv2.resize(face_section, (100, 100))

        # predict the face section
        ypred = knn.predict( X_train, Y_train, face_section.reshape(1,-1), k=3)
        print(f"label: {ypred}, name: {map[ypred]}")
        
    cv2.putText(image, map[ypred], (x,y-padding), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.rectangle(image, (x-padding,y-padding), (x+w+padding, y+h+padding), (0, 0,255), 2)
    cv2.imshow("Captured image", image)

    # exit out of the loop
    key_pressed = cv2.waitKey(25)
    if key_pressed == ord('q'):
        print(key_pressed)
        print("Q was pressed")
        break

cap.release()
cv2.destroyAllWindows()



