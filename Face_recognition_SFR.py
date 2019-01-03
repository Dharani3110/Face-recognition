# import the libraries
import os
import cv2
import face_recognition
import time
import numpy as np
#from tempfile import TemporaryFile


# make a list of all the available images
images = os.listdir('/home/soliton/work/SFR/dev/SFR_code/sample_train_images/')


#view the test image
'''
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',image_to_be_matched)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(face_recognition.face_locations(image_to_be_matched))
'''

#print("\nFEATURES:",face_recognition.face_landmarks(image_to_be_matched)[0])
cap = cv2.VideoCapture(-1)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
distances = []
current_image_encoded=[]

def recognise_face(faces, test_image, current_image_encoded):

    print("In Detect face.....")
    for (x, y, w, h) in faces:
        cv2.rectangle(test_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    print(len(faces))
    cv2.imshow("face_detect", test_image)
    roi_color = test_image[y:y + h, x:x + w]
    image_to_be_matched = face_recognition.load_image_file(roi_color)

        # encoded the loaded image into a feature vector
    image_to_be_matched_encoded = face_recognition.face_encodings(image_to_be_matched)[0]

    for current_image_encoding in current_image_encoded:
        # compute distance between test image and train image
        distance = face_recognition.face_distance([image_to_be_matched_encoded], current_image_encoding[0])
        distances.append([distance[0], current_image_encoding[1]])
    #
    # # print(time.time() - start)
    #
    # #np.save(s,current_image_encoded )
    # # match your image with the image and check if it matches
    # #result = face_recognition.compare_faces( [image_to_be_matched_encoded], current_image_encoded , tolerance=0.67)

    distances.sort(key=lambda x: x[0])
    # print("\nTest image:",test_image)
    print(distances[0])

#compute the encodings for all train_images
for image in images:
    (image_name,ext)= os.path.splitext(image)
    # load the image
    train_img = face_recognition.load_image_file("sample_train_images/" + image)
    # encode the loaded image into a feature vector
    current_image_encoded.append([face_recognition.face_encodings(train_img)[0], image_name])

# start = time.time()
while True:
    # load your image
    ret, test_image = cap.read()
    if ret:
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        cv2.imshow("test_image", test_image)
        if cv2.waitKey(30) & 0xff == ord('q'):
            break
        if (len(faces) != 0):
            recognise_face(faces, test_image, current_image_encoded)
        else:
            continue
    else:
        print("NO FRAMES.....")
        break





cap.release()
cv2.destroyAllWindows()
#[print(each_image_distance) for each_image_distance in distances]

