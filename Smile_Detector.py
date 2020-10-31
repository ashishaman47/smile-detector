import cv2

from random import randrange

# Face Classifier
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Smile detector --> works well with teeth smile
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

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

    # print(faces)

    # iterate over each of the faces and draw rectangle
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 4)

        # slicing to face portion --> (z,y,x)
        the_face = frame[y:y+h, x:x+w]

        # change the cropped portion of face to grayscale
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        # detect smile
        # scale factor is how much u wanna blur the image --> bluring will make it easier to detect the facial feature rathere than other side ways feature
        # minNeighbors is total min how many rectangles in the area that it will count as smile --> if we reduce this no. to 1 it will pick only 1 smile
        # btw these two no. are close to smile
        smiles = smile_detector.detectMultiScale(
            face_grayscale, scaleFactor=1.7, minNeighbors=20)

        # Find all smile in the face
        # for (x_, y_, w_, h_) in smiles:

        #     # Draw a rectangle around the smile
        #     cv2.rectangle(the_face, (x_, y_), (x_ + w_, y_ + h_),
        #                   (50, 50, 200), 4)

        # Label the face as smiling instead of drawing rectangle over smile
        if len(smiles) > 0:
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=3,
                        fontFace=cv2.FONT_HERSHEY_PLAIN, color=(randrange(256), randrange(256), randrange(256)))

    # Display
    cv2.imshow('Why So Serious??', frame)

    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

# cleanup
webcam.release()
cv2.destroyAllWindows()


print('Code Completed')
