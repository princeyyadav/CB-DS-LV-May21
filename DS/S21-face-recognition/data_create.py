import numpy as np
import cv2
import sys

# read the image

filename = sys.argv[1]
img = cv2.imread(filename)

img = cv2.resize(img, (150, 150))
print(img.shape)

img2 = img.flatten()

X = []
for i in range(30):
    X.append(img2)

X = np.array(X)
print(X.shape)

np.save(filename.split(".")[0], X)


while True:

    cv2.imshow("Hugh", img)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cv2.destroyAllWindows()



