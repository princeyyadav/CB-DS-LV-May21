import numpy as np
import cv2


cap = cv2.VideoCapture(0) # object to interact with web cam

face_cascade = cv2.CascadeClassifier("../S21-face-recognition/haarcascade_frontalface_alt.xml")

X = []

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
        # cv2.rectangle(image, (x-padding,y-padding), (x+w+padding, y+h+padding), (0, 0,255), 2) # draw rectange on an image (images, start pt, end pt, color, thickness)

        # slice the face out of the image
        face_section = image[ y-padding:y+h+padding, x-padding:x+w+padding ]
        face_section = cv2.resize(face_section, (100, 100))
        
    # cv2.imshow("Captured image", image)
    cv2.imshow("face section", face_section)
    # print(face_section.shape)

    # append the face section to my X list
    X.append(face_section.reshape(1,-1)) # rather use flatten() or reshape(-1,)

    print(len(X), X[-1].shape)

    # exit out of the loop
    key_pressed = cv2.waitKey(25)
    # print(key_pressed, type(key_pressed))
    if key_pressed == ord('q'):
        print(key_pressed)
        print("Q was pressed")
        break

cap.release()
cv2.destroyAllWindows()

# convert the X list to numpy array
X = np.array(X)
filename = input("Enter the name of person: ")
np.save(filename, X)


