import cv2
import sys

face_cascPath = 'haarcascade_frontalface_default.xml'
eye_Cascade = 'haarcascade_eye.xml'
faceCascade = cv2.CascadeClassifier(face_cascPath)
eyeCascade = cv2.CascadeClassifier(eye_Cascade)

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    print("Found {0} Faces!".format(len(faces)))

    if len(faces) > 0:
        status = cv2.imwrite('faces_detected.jpg', frame)
        print("[INFO] Image faces_detected.jpg written to filesystem: ", status)

    # Loop Over Faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the faces
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Draw a circle around the faces

        center = (x + w // 2, y + h // 2)
        radius = (w + h) // 4
        img = cv2.circle(frame, center, radius, (255, 0, 0), 2)

        # Write Face to local file system

        face_color = frame[y:y + h, x:x + w]
        print("[INFO] Object found. Saving locally.")
        cv2.imwrite(str(w) + str(h) + '_faces.jpg', face_color)

        face_gray = img[y:y + h, x:x + w]
        eyes = eyeCascade.detectMultiScale(face_gray, 1.2, 3)
        for (ex, ey, ew, eh) in eyes:
            eye_center = (ex + ew // 2, ey + eh // 2)
            eye_radius = (ew + eh) // 4
            cv2.circle(face_color, eye_center, eye_radius, (128, 128, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face & Eye Detection', frame)

    # Frame will be close after ESC key is clicked
    if cv2.waitKey(1) & 0xFF == 27:
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyWindow('Face & Eye Detection')
sys.exit(0)
