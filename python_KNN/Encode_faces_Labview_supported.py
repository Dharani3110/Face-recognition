import os
import dlib
import glob
import cv2
import numpy as np
import pickle
import csv
from imutils import paths
import pandas as pd
import sklearn.neighbors.typedefs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib

predictor_path = 'dependencies\shape_predictor_5_face_landmarks.dat'
face_rec_model_path = 'dependencies\dlib_face_recognition_resnet_model_v1.dat'

# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face, and finally the
# face recognition model.
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

#win = dlib.image_window()
data = []

def list_to_csv(data):
    """
    This function converts encodings to CSV file.
    :param data: It contains list of names and encodings
    :return: it doesn't return anything
    """
    # sorting the encoding according to alphabetical order.
    data.sort(key=lambda x: x[0])
    names_list = []
    with open('dependencies\Encodings_csv.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, dialect='excel')
        for encoding in data:
            # converting the 128-d encoding to string
            encoding_list = [str(value) for value in encoding[1]]
            # converting list of string to list of float value
            encoding_list_value = [float(value) for value in encoding_list]
            # converting the name to a list.
            names_list.append(encoding[0])
            row = names_list+encoding_list_value
            writer.writerow(row)
            names_list.pop()

def store_KNN_model():
    # read the csv file containing the encodings
    df = pd.read_csv('dependencies\Encodings_csv.csv', header=None)
    # separate the encodings from the csv file
    encodings = df.drop(df.columns[0], axis=1)
    # separate the class name i.e name of person from the csv file
    names = df[0]
    # specify number of neighbours for tthe model
    knn = KNeighborsClassifier(n_neighbors=5)
    # Train the model
    knn.fit(encodings, names)
    filename = 'dependencies\knn_classifier_model.sav'
    # Store the model for later use
    joblib.dump(knn, filename)
    print("\nKNN Model trained and stored....\n")

# Now process all the images
def create_encodings():
    print("Preparing database............")
    image_paths = list(paths.list_images("train_images"))
    for (i, image)  in enumerate(image_paths):
        print("Processing image: {0}/{1}".format(i+1,len(image_paths)))
        img =cv2.imread(image)
        (img_folder, ext) = os.path.splitext(image)
        (main_fldr , fldr_name, img_name )= img_folder.split('\\')
        '''img = cv2.resize(img, (224, 224))
        win.clear_overlay()
        win.set_image(img)'''

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = detector(img, 1)
        if len(dets) != 0:
            # Now process each face found.
            for k, d in enumerate(dets):
                # Get the landmarks/parts for the face in box d.
                shape = sp(img, d)
                # Draw the face landmarks on the screen so we can see what face is currently being processed.
                '''win.clear_overlay()
                win.add_overlay(d)
                win.add_overlay(shape)'''

                # Compute the 128D vector that describes the face in img identified by
                face_descriptor = list(facerec.compute_face_descriptor(img, shape))
                # Append the encoding along with its corresponding image name to a list
                data.append([fldr_name, np.array(face_descriptor)])

    # convert the encodings into csv file
    list_to_csv(data)
    print("\nCompleted encoding...")
    with open('dependencies\Encodings', 'wb') as fp:
        fp.write(pickle.dumps(data))
    store_KNN_model()


if __name__ == '__main__':
    create_encodings()
