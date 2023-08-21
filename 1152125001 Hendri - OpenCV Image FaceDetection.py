import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# IMG SOURCE
srcpath = "./20220828_101502.jpg"
img = Image.open(srcpath)
img_arr = np.asarray(img)   # CONVERT IMG2ARR


# FIGURE: CANVAS SIZE
plt.figure(figsize=(6, 3))

# IMG2SHOW 1
plt.subplot(1, 2, 1)
plt.imshow(img_arr[:, :, 0], cmap='gray')
plt.title("Grayscale(ch Red)")


# START OF HCC FACE & EYE (MODEL)
# 4Face
img_facecascade = cv2.CascadeClassifier(
    "./harcascade/haarcascade_frontalface_alt.xml")
# 4Eye
img_eyecascade = cv2.CascadeClassifier(
    "./harcascade/haarcascade_eye.xml")
# END OF HCC FACE & EYE


def detect_faceandeyev2(img2detect):   # FACE&EYE MARKING FUNCT
    face_img = img2detect.copy()
    face_box = img_facecascade.detectMultiScale(
        face_img, scaleFactor=1.2, minNeighbors=5)
    eye_box = img_eyecascade.detectMultiScale(
        face_img, scaleFactor=1.2, minNeighbors=5)

    for (x, y, l, t) in face_box:
        cv2.rectangle(face_img, (x, y), (x+l, y+t), (255, 255, 255), 10)
    for (x, y, l, t) in eye_box:
        cv2.rectangle(face_img, (x, y), (x+l, y+t), (255, 255, 255), 7)
    return face_img


# SET OUTPUT
output = detect_faceandeyev2(img_arr)
# OUTPUT: IMG2SHOW 2
plt.subplot(1, 2, 2)
plt.imshow(output)
plt.title("Output Face&Eye V2(RGB)")


# FIGURE: CANVAS FOOTER
plt.tight_layout()
plt.show()
