"""
SOLITON FACE RECOGNITION
It is to recognize all the employees of Soliton using one-shot learning technique.

"""
import os
import cv2
import face_recognition
import time
import numpy as np
from PIL import Image
import screeninfo
import argparse
import imutils
import pickle
import cv2

#Get encodings path as argument
obj = argparse.ArgumentParser()
#Input (python_filename -e path_to_encodings )as argument
obj.add_argument("-e1", "--encodings_large", required=True,help="path to serialized db of facial encodings")
#obj.add_argument("-e2", "--encodings_small", required=True,help="path to serialized db of facial encodings")
args = vars(obj.parse_args())
#Read the encodings from the path_to_encodings provided above
data= pickle.loads(open(args["encodings_large"], "rb").read())
train_images_encodings_large=data
#data= pickle.loads(open(args["encodings_small"], "rb").read())
#train_images_encodings_small=data



def recognize_face(img, model, train_images_encodings):
    """
    It is to recognize the faces in the passed cropped images.
    :param img: pass the image to be recognized
    :return: returns "train_image_name" if the img_encodings match with any of the train_image_encodings
             returns "Unknown"-if the image is not found in database
    """
    try:
      #print(face_recognition.face_locations(img))
      #Encode the cropped image
      img_encodings = face_recognition.face_encodings(img,None,1,model)
      for img_encoding in img_encodings:
          match_results = []
          for train_image_encodings in train_images_encodings:
             match_results.append(face_recognition.compare_faces(train_image_encodings[0], img_encoding)[0])
          name = "Unknown"
          distances = []
          if True in match_results:

              for train_image_encodings in train_images_encodings:
                distance = face_recognition.face_distance(img_encoding, train_image_encodings[0])
                if (distance[0]<0.55):
                 distances.append([distance[0], train_image_encodings[1]])
              print('\n')
              distances.sort(key=lambda x: x[0])
              print("model:",model)
              [ print(distances[i]) for i in [0,1,2,3,4]]
              name = distances[0][1]

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
            res = np.hstack((gray, equ))

            #Obtain locations of the detected faces using haarcascade
            faces = face_cascade.detectMultiScale(res, 1.3, 5)
            for (x, y, w, h) in faces:
                #Draw boxes once the faces are detected
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

                if (len(faces) != 0):
                    #crop the frame using the locations from haarcascade detection
                    #crop_img = img[y - constant :y + h + constant, x - constant:x + w - constant]
                    crop_img = img[y:y + h, x:x + w]
                    #pass the cropped image to function recognize()
                    #start_time=time.time()
                    text_for_large = recognize_face(crop_img,"large",train_images_encodings_large)
                    #if text_for_large is not None:
                    # print("Time taken for recognition(model-large):\t{0}".format((time.time()-start_time)*1000))

                    '''start_time = time.time()
                    text_for_small = recognize_face(crop_img, "small", train_images_encodings_small)
                    if text_for_small is not None:
                        print("Time taken for recognition(model-small):\t{0}".format((time.time() - start_time) * 1000))'''

                    #create a filled box above the previous box and display the name recognized
                    cv2.rectangle(img, (x, y - 35), (x + w, y), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, text_for_large, (x+6, y-6),cv2.FONT_HERSHEY_DUPLEX ,0.5, (255, 255, 255), 1)

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
