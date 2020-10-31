import cv2

# Face Classifier
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')

# Grab a webcam feed
webcam = cv2.VideoCapture(0)

while True:
    # Read current frame from webcam
    successful_frame_read, frame = webcam.read()

    # if ther's an error, abort
    if not successful_frame_read:
        break

    # change to grayscale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(frame_grayscale, scaleFactor=1.7)

    # iterate over each of the faces and draw rectangle
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 4)

        # slicing to face portion --> (z,y,x)
        the_face = frame[y:y+h, x:x+w]

        # change the cropped portion of face to grayscale
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        # Detect Eyes
        eyes = eye_detector.detectMultiScale(face_grayscale, scaleFactor=1.7)

        # Find eyes on face
        for (x_, y_, w_, h_) in eyes:
            # Draw rectangle around eyes
            cv2.rectangle(the_face, (x_, y_),
                          (x_ + w_, y_ + h_), (255, 0, 0), 4)

    # Display
    cv2.imshow('Eyes >>', frame)

    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

# cleanup
webcam.release()
cv2.destroyAllWindows()


print('Code Completed')
