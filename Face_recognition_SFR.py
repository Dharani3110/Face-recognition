"""
SOLITON FACE RECOGNITION
It is to recognize all the employees of Soliton using one-shot learning technique.

"""

import face_recognition
import time
import numpy as np
import screeninfo
import argparse
import pickle
import cv2

#Get encodings path as argument
obj = argparse.ArgumentParser()
#Input (python_filename -e path_to_encodings )as argument
obj.add_argument("-e", "--encodings", required=True,help="path to serialized db of facial encodings")
args = vars(obj.parse_args())

#Read the encodings from the path_to_encodings provided above
data = pickle.loads(open(args["encodings"], "rb").read())
train_images_encodings = data
min_difference = 0.001
tolerance = 0.48
min_distance = 0.55


def detect_resemblance(distances):
    """
    It is to overcome the effect of close resemblance between two persons
    :param distances:pass the list of distances with distance less than 0.55
    :return: returns the folder name of the first_image
               1.if the first and second images are very similar and belong to same folder
               2.if the first and second images are not very similar
             returns None if the images are of different folders though they are very similar
    """
    (first, second) = (distances[i] for i in [0, 1])
    diff = second[0] - first[0]
    # print(diff)
    if (diff  <  min_difference):
        (first_folder,first_img)=first[1].split('/')
        (second_folder, second_img) = second[1].split('/')
        if (first_folder != second_folder):
          return None
        elif (first_folder == second_folder):
          return first_folder
    else:
          (folder_name, img_name) = distances[0][1].split('/')
          return folder_name


def recognize_face(img, model, train_images_encodings):
    """
    It is to recognize the faces in the passed cropped images.
    :param img: pass the image to be recognized
    :return: returns "train_image_name" if the img_encodings match with any of the train_image_encodings
             returns "Unknown"-if the image is not found in database
    """
    try:
      #Encode the cropped image
       img_encodings = face_recognition.face_encodings(img,None,1,model)
       for img_encoding in img_encodings:
          match_results = []
          for train_image_encodings in train_images_encodings:
             match_results.append(face_recognition.compare_faces(train_image_encodings[0], img_encoding,tolerance)[0])
          name = None
          distances = []

          if True in match_results:

              for train_image_encodings in train_images_encodings:
                distance = face_recognition.face_distance(img_encoding, train_image_encodings[0])
                if (distance[0] < min_distance):
                 distances.append([distance[0], train_image_encodings[1]])
              distances.sort(key = lambda x: x[0])
              name = detect_resemblance(distances)


       return name

    except:
       return None


def main():
    """
    It is the main function which calls the functions for detection and recognition.
    :return: returns full-screen video display with boxes bounding the detected faces and names of the recognized faces
    """
    print("------------------ Soliton Face Recognition ------------------\n\n")

    #get screen info
    screen_detail = screeninfo.get_monitors()[0]

    # Get a reference to webcam
    cap = cv2.VideoCapture(-1)
    #cap.set(cv2.CAP_PROP_POS_MSEC,30)

    #Explain what this xml file is
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #fps = cap.get(cv2.CAP_PROP_FPS)
    #print("Frames per second of webcam is: {0}".format(fps))

    print("Look into camera!")
    #constant=np.int32(20)
    while True:
        # Grab a single frame of video
        ret, test_image = cap.read()

        img = test_image
        if ret:
            #Preprocess the captured frame for better recognition under dark conditions
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            equ = cv2.equalizeHist(gray)
            #res = np.hstack((gray, equ))

            #Obtain locations of the detected faces using haarcascade
            faces = face_cascade.detectMultiScale(equ, 1.3, 5)
            for (x, y, w, h) in faces:
                #Draw boxes once the faces are detected
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

                if (len(faces) != 0):
                    #crop the frame using the locations from haarcascade detection
                    #crop_img = img[y - constant :y + h + constant, x - constant:x + w - constant]
                    crop_img = img[y:y + h, x:x + w]
                    #pass the cropped image to function recognize()
                    #start_time=time.time()
                    text= recognize_face(crop_img,"large",train_images_encodings)
                    #if text is not None:
                    # print("Time taken for recognition:\t{0}".format((time.time()-start_time)*1000))

                    #create a filled box above the previous box and display the name recognized
                    cv2.rectangle(img, (x, y - 35), (x + w, y), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, text, (x+6, y-6),cv2.FONT_HERSHEY_DUPLEX ,0.5, (255, 255, 255), 1)

            window_name = 'Find Your Face :P'
            #Results are displayed in full screen
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.moveWindow(window_name, screen_detail.x - 1, screen_detail.y - 1)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow(window_name, img)

            #Stop the execution of the program by pressing 'q' key
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
            else:
                continue
    #Close all windows
    cap.release()
    cv2.destroyAllWindows()

main()
