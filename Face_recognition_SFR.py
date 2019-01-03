"""
SOLITON FACE RECOGNITION
Comment here

"""
import os
import cv2
import face_recognition
import time
import numpy as np

def recognise_face(test_image, train_images_encoded):
    """
    Comment here
    :param faces:
    :param test_image:
    :param current_image_encoded:
    :return:
    """
    cv2.imshow("test_image", test_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    image_to_be_matched = face_recognition.load_image_file(test_image)
    image_to_be_matched_encoded = face_recognition.face_encodings(image_to_be_matched)[0]
    print("\n\n\n\n",image_to_be_matched_encoded)

    for train_image_encoded in train_images_encoded:
        # compute distance between test image and train image
        distance = face_recognition.face_distance([image_to_be_matched_encoded], train_image_encoded[0])
        distances.append([distance[0], train_image_encoded[1]])
    # print(time.time() - start)
    #
    # #np.save(s,current_image_encoded )
    # # match your image with the image and check if it matches
    # #result = face_recognition.compare_faces( [image_to_be_matched_encoded], current_image_encoded , tolerance=0.67)
    distances.sort(key=lambda x: x[0])
   # print("\nTest image:",test_image)
    return distances

def main():
    """
    Comment here
    :return:
    """
    print("------------------ Soliton Face Recognition ------------------\n\n")
    #Make the path relative
    train_image_encoded = []
    images = os.listdir('/home/soliton/work/SFR/dev/SFR_code/sample_train_images/')
    '''
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image',image_to_be_matched)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(face_recognition.face_locations(image_to_be_matched))
    '''
    cap = cv2.VideoCapture(-1)
    #Explain what this xml file is
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #compute the encodings for all train_images
    for image in images:
        (image_name,ext)= os.path.splitext(image)
        # load the image
        train_img = face_recognition.load_image_file("sample_train_images/" + image)
        # encode the loaded image into a feature vector
        train_image_encoded.append([face_recognition.face_encodings(train_img)[0], image_name])

    # start = time.time()
    while True:
        # load your image
        ret, test_image = cap.read()
        img = test_image
        if ret:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            equ = cv2.equalizeHist(gray)
            faces = face_cascade.detectMultiScale(img, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow("test_image", img)
            if cv2.waitKey(30) & 0xff == ord('q'):
                break
            if (len(faces) != 0):
                recognise_face(test_image, train_image_encoded)
            else:
                continue

    cap.release()
    cv2.destroyAllWindows()
    #[print(each_image_distance) for each_image_distance in distances]
main()
