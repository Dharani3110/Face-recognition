import os
import cv2
import face_recognition
from PIL import Image
from imutils import paths
import argparse
import pickle

#Get encodings path and dataset path as arguments
obj = argparse.ArgumentParser()
obj.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
obj.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
args = vars(obj.parse_args())

imagePaths = list(paths.list_images(args["dataset"]))
data=[]
for (i, imagePath) in enumerate(imagePaths):

    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
    (img_name, ext) = os.path.splitext(imagePath)
    (folder, name) = img_name.split("/")

    # load the input image and convert it from RGB (OpenCV ordering)
    # to dlib ordering (RGB)
    image = cv2.imread(imagePath)
    #rgb= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # compute the facial embedding for the face
    encoding = face_recognition.face_encodings(image)
    data.append([encoding,name])
# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")

f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()
