# import the libraries
import os
import cv2
import face_recognition
import time
import numpy as np
#from tempfile import TemporaryFile


# make a list of all the available images
images = os.listdir('/home/soliton/work/SFR/dev/SFR_code/train_images/')


#view the test image
'''
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',image_to_be_matched)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(face_recognition.face_locations(image_to_be_matched))
'''

#print("\nFEATURES:",face_recognition.face_landmarks(image_to_be_matched)[0])


distances = []
current_image_encoded=[]

#compute the encodings for all train_images
for image in images:
    (image_name,ext)= os.path.splitext(image)
    # load the image
    train_img = face_recognition.load_image_file("train_images/" + image)
    # encode the loaded image into a feature vector
    current_image_encoded.append([face_recognition.face_encodings(train_img)[0], image_name])

start = time.time()

# load your image
test_image = 'test_images/ani_test.jpg'
image_to_be_matched = face_recognition.load_image_file(test_image)

# encoded the loaded image into a feature vector
image_to_be_matched_encoded = face_recognition.face_encodings(image_to_be_matched)[0]


for current_image_encoding in current_image_encoded:
    #compute distance between test image and train image
    distance = face_recognition.face_distance([image_to_be_matched_encoded], current_image_encoding[0])
    distances.append([distance[0], current_image_encoding[1]])

print(time.time() - start)

    #np.save(s,current_image_encoded )
    # match your image with the image and check if it matches
    #result = face_recognition.compare_faces( [image_to_be_matched_encoded], current_image_encoded , tolerance=0.67)

distances.sort(key=lambda x: x[0])
print("\nTest image:",test_image)
print(distances[0])

#[print(each_image_distance) for each_image_distance in distances]

