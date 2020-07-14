# Face, eyes and Smile Detection

# Importing the libraries
import cv2

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Function takes image in b/w(gray) and original image(frame)
# and return image with detector rectangles.
def detect(gray, frame):
    
    #face
    #parameters of functions (GreyImage, ScalingFactorOfImage, NoOfNeighbourZoneFacesNearThisFace)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        #parameters (OrginalImage,(TOpLefCorner), (botthomRightCorner),(RGB),BorderWidth)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        

        #Rectangular Area of Face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        

        #smile (find smile only inside the rectangular area of face)
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
    
    
        #eyes (find eyes only inside the rectangular area of face)
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    
    return frame

# Doing some Face Recognition with the webcam
# 0->WebCam, 1->externalCam
video_capture = cv2.VideoCapture(0)

while True:

    #To ignore first item, we can use '_',warna var ko naam dena pdta
    #Since video_capture.read() return two values

    #frame contains the last frame
    _, frame = video_capture.read()
    
    #convert to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #repaint with apropriate rectangles
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)

    #Stop camera on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# We turn the webcam off.
# We destroy all the windows inside which the images were displayed.
video_capture.release()
cv2.destroyAllWindows()