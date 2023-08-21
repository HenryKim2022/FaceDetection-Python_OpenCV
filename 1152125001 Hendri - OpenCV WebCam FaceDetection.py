import cv2

# START OF HCC FACE & EYE(MODEL)
# 4Face
img_facecascade = cv2.CascadeClassifier(
    "./harcascade/haarcascade_frontalface_alt.xml")
# 4Eye
img_eyecascade = cv2.CascadeClassifier(
    "./harcascade/haarcascade_eye.xml")
# END OF HCC FACE & EYE


def detect_faceandeyev2(img2detect):   # FACE&EYE MARKING FUNCT []
    face_img = img2detect.copy()
    face_box = img_facecascade.detectMultiScale(
        face_img, scaleFactor=1.2, minNeighbors=5)
    eye_box = img_eyecascade.detectMultiScale(
        face_img, scaleFactor=1.2, minNeighbors=5)

    for (x, y, l, t) in face_box:
        cv2.rectangle(face_img, (x, y), (x+l, y+t), (0, 255, 0), 6)  # GREEN
    for (x, y, l, t) in eye_box:
        cv2.rectangle(face_img, (x, y), (x+l, y+t),
                      (255, 255, 255), 4)  # WHITE
    return face_img


# WEBCAM
# Note: Change 0 to use which webcam
capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
capture.set(cv2.CAP_PROP_FPS, 280)  # Frame rate/second(Lagness) --> 1-300fps


def denoise_frame(frame):
    # START OF FILTER
    # Apply Non-Local Means Denoising
    denoised_frame = cv2.fastNlMeansDenoisingColored(
        frame, None, 3, 3, 3, 6)

    # Adjust contrast and brightness
    # Contrast control (1.0 - 3.0, higher values increase contrast)
    alpha = 0.9
    # Brightness control (0-100, higher values increase brightness)
    beta = 2
    adjusted_frame = cv2.convertScaleAbs(
        denoised_frame, alpha=alpha, beta=beta)

    return adjusted_frame
   # END OF FILTER


# ALWAYS UPDATING FRAME(AUF's)
while True:
    ret, frame = capture.read()
    frame = detect_faceandeyev2(frame)
    frame = cv2.resize(frame, (400, 235))   # Resize Dimension

    denoised_frame = denoise_frame(frame)   # Apply Denoising
    cv2.imshow("WebCam Face & Eye Detection", denoised_frame)

    if (cv2.waitKey(1) & 0xFF == 27):   # ESC keyboard key
        break


# SHOWING OUTPUT & KILL ALL WINDOWS(cv2)
capture.release()
cv2.destroyAllWindows()
