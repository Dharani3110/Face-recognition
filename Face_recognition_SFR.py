"""
SOLITON FACE RECOGNITION
Comment here

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
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
                    help="path to serialized db of facial encodings")
args = vars(ap.parse_args())
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

def recognize_face(img):
    """
    Comment here
    :param faces:
    :param test_image:
    :param current_image_encoded:

    :return:
    """

    encodings = face_recognition.face_encodings(img)
    try:
     for encoding in encodings:
        match_results=[]
        for datum in data:
          match_results.append(face_recognition.compare_faces(datum[0],encoding)[0])
        name="Unknown"
        distances=[]
        if True in match_results:
            
            for datum in data:
                distance=face_recognition.face_distance(encoding,datum[0])
                distances.append([distance[0],datum[1]])
            distances.sort(key=lambda x: x[0])
            name=distances[0][1]
    #names.append(name)
     return name

    except:
       pass



def main():
    """
    Comment here
    :return:
    """
    print("------------------ Soliton Face Recognition ------------------\n\n")

    #get screen info
    screen_detail = screeninfo.get_monitors()[0]

    # Get a reference to webcam
    cap = cv2.VideoCapture(-1)
    cap.set(cv2.CAP_PROP_POS_MSEC,30)

    #Explain what this xml file is
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Frames per second of webcam is: {0}".format(fps))

    print("Look into camera!")
    e=np.int32(20)
    count = 1
    while True:
        # Grab a single frame of video
        ret, test_image = cap.read()

        img = test_image
        if ret:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            equ = cv2.equalizeHist(gray)
            res = np.hstack((gray, equ))


            faces = face_cascade.detectMultiScale(res, 1.3, 5)
            for (x, y, w, h) in faces:
                #print(x, y, w, h,x - e,x + w + e, y - e,y + h + e)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

                if (len(faces) != 0):
                    #crop_img = img[y - e:y + h + e, x - e:x + w + e]
                    crop_img = img[y :y + h, x:x + w]
                    # box = (x, y, w, h)
                    # crop1 = Image.crop(box)
                    #im_pil = Image.fromarray(crop_img)
                    #im_pil.save('crop.png')
                    # cv2.imwrite('crop' + str(count) + '.', crop_img)
                    #count += 1
                    #cv2.imshow('crop', cv2.resize(crop_img,(110,110)))
                    #cv2.waitKey(0)
                    text = recognize_face(crop_img)
                    cv2.rectangle(img, (x, y - 35), (x + w, y), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, text, (x+6, y-6),cv2.FONT_HERSHEY_DUPLEX ,0.5, (255, 255, 255), 1)

            window_name = 'Find Your Face :P'
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.moveWindow(window_name, screen_detail.x - 1, screen_detail.y - 1)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow(window_name, img)

            if cv2.waitKey(1) & 0xff == ord('q'):
                break
            else:
                continue

    cap.release()
    cv2.destroyAllWindows()

main()
