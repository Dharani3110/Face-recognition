import pandas as pd
from sklearn.externals import joblib
import cv2
import dlib
import pickle
import numpy as np
import time
import distutils
import sklearn.neighbors.typedefs
import xml.etree.ElementTree as xml
from imutils import paths
import configparser

# Load the Knn model
knn = joblib.load('dependencies\knn_classifier_model.sav')

# load the encodings we have created
data = pickle.loads(open('dependencies\Encodings', 'rb').read())
print("\nNumber of encodings loaded:  ",len(data))

# Store face detector model, shape predictor model and face recognition model in individual variable.
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('dependencies\shape_predictor_5_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1(
    'dependencies\dlib_face_recognition_resnet_model_v1.dat')

frame_rect_color = (0, 255, 0)
frame_text_color = (0, 0, 0)

# Tweak this parameter to get minimised false recognition
config = configparser.ConfigParser()
config.read('dependencies\config.ini')
min_distance = config.getfloat("Threshold_initialization",'min_distance')
print("\nInitialized threshold: ",min_distance)


def read_image(filename):
    """
    This function reads the image specified by its filename
    :param filename: It is string containing the filename along with the extension of image
    :return: It returns image stored in the specified filename
    """
    image = cv2.imread(filename, 1)
    return image



def draw_rectangle(image, coordinates_1, coordinates_2, rect_color, filled_color):
    """
    This function draws rectangle on the image passed and according to the coordinates specified.
    :param image: Image on which we need to draw a rectangle
    :param coordinates_1: It is a tuple containing value of left, top points of rectangle
    :param coordinates_2: It is a tuple containing value of right, bottom points of rectangle
    :param rect_color: It is a tuple specifying colour of the rectangle
    :param filled_color: It is a boolean value used to either fill the rectangle with color or not.
    :return: It returns image with drawn rectangle
    """
    if filled_color:
        cv2.rectangle(image, coordinates_1, coordinates_2,
                      rect_color, cv2.FILLED)
    else:
        cv2.rectangle(image, coordinates_1, coordinates_2, rect_color, 1)
    return image


def write_text(image, message, coord, text_color):
    """

    :param image: Image on which we need to write a text
    :param message: String containing text to be written on the image
    :param coord: It is tuple containing coordinates from where the text has to be written
    :param text_color: It is tuple specifying color of the text
    :return: It returns image with text written on it.
    """
    cv2.putText(image, message, coord,
                cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color, 1)
    return image


def L2_distance(face_encoding, index_list):
    """
    This function calculates the L2 distance between each of the encodings of N Neighbours with the encodings of the
    face detected.

    :param face_encoding: It gets a 128-dimensional list of facial encoding of face detected in the test_image.
    :param index_list: It contains the index values of n nearest neighbours returned by the knn classifier.
    :return: A string indicating the name of the person recognised based on the threshold(min_distance) we set.
    """
    database_list = [tuple([data[index][0], np.linalg.norm(
        face_encoding - data[index][1])]) for index in index_list]
    database_list.sort(key=lambda x: x[1])
    #[print(i) for i in database_list]
    print('\n')
    if database_list[0][1] > min_distance:
        duplicate = list(database_list[0])
        duplicate[0] = 'Unknown' # If no distance is less than the min distance, name is considered as 'Unknown".
        #duplicate[0] = None
        database_list.insert(0, tuple(duplicate))
    return database_list


def face_recogniser(frame):
    """
    This function detects and recognises faces in the given image.
    :return: It returns a .xml file with names and the locations of the persons detected.
    """
    start = time.time()
    faces = detector(frame, 1)
    data = xml.Element('faces')
    xml.SubElement(data, "faces_detected").text = str(len(faces))
    if len(faces) != 0:
        for face, d in enumerate(faces):

            # Get the locations of facial features like eyes, nose for using it to create encodings
            shape = sp(frame, d)

            # Get the coordinates of bounding box enclosing the face.
            left = d.left()
            top = d.top()
            right = d.right()
            bottom = d.bottom()

            # Calculate encodings of the face detected
            start_encode = time.time()
            face_descriptor = list(
                face_recognition_model.compute_face_descriptor(frame, shape))
            print('\n')
            print("Time taken to encode face "+ str(face+1) + " :::  " +
                  str((time.time()-start_encode) * 1000)+"  ms")

            face_encoding = pd.DataFrame([face_descriptor])
            face_encoding_list = [np.array(face_descriptor)]

            # Get indices the N Neighbours of the facial encoding
            list_neighbors = knn.kneighbors(
                face_encoding, return_distance=False)

            # Calculate the L2 distance between the encodings of N neighbours and the detected face.
            start_compare = time.time()
            database_list = L2_distance(
                face_encoding_list, list_neighbors[0])
            print("   Time taken to compare with N neighbours:::  " +
                  str((time.time()-start_compare) * 1000)+"  ms")


            print("   Time taken for detection and recognition :::  " +
                  str((time.time()-start) * 1000)+"  ms")
            person_name = database_list[0][0]

            #XML creation
            face_id = "face" + str(face+1)
            name_id = "name" + str(face+1)
            face_id = xml.SubElement(data, face_id)

            name_id = xml.SubElement(face_id, name_id)
            name_id.text = person_name

            locations = xml.SubElement(face_id, "locations")
            left_loc = xml.SubElement(locations, "left")
            right_loc = xml.SubElement(locations, "right")
            top_loc = xml.SubElement(locations, "top")
            bottom_loc = xml.SubElement(locations, "bottom")

            left_loc.text = str(left)
            right_loc.text = str(right)
            top_loc.text = str(top)
            bottom_loc.text = str(bottom)

    #Writing to an .xml file
    mydata = xml.tostring(data , encoding="unicode")
    myfile = open("Recognition_results.xml", "w")
    myfile.write(mydata)
    print('\nResults written to Recognition_results.xml...\n')

if __name__ == '__main__':
    image_paths = list(paths.list_images("test_image"))
    print("\nTest image:", image_paths[0])
    image = cv2.imread(image_paths[0])
    face_recogniser(image)
