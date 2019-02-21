import os
import cv2
import face_recognition
from PIL import Image
from imutils import paths
import argparse
import pickle

#Get dataset path and encodings path as arguments
obj = argparse.ArgumentParser()
#Input (python_filename -i path_to_dataset )as argument
obj.add_argument("-i", "--dataset", required=True,help="path to input directory of faces + images")
#Input (python_filename -e path_to_encodings )as argument
obj.add_argument("-e", "--encodings", required=True,help="path to serialized db of facial encodings")
args = vars(obj.parse_args())

#Get the paths of the images as a list
imagePaths = list(paths.list_images(args["dataset"]))
data=[]
for (i, imagePath) in enumerate(imagePaths):
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    # extract the person name with folder name from the image path
    (img_name, ext) = os.path.splitext(imagePath)
    (main_folder,folder,image_name) = img_name.split("/")
    name=folder+"/"+image_name
    image = cv2.imread(imagePath)
    # compute the facial embedding for the image
    encoding = face_recognition.face_encodings(image)
    data.append([encoding,name])
# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")

f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()

