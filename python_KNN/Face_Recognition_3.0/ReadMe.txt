INSTRUCTIONS

***************************** Setting the threshold for face recognition **********************************************************************

1. Open "config" file inside "dependencies" folder.
2. The default value is min_distance = 0.46.
3. Save the "config" file, once the value of the min_distance is changed.


***************************** Database creation ***********************************************************************************************

1. To include a new image in database, create a folder with name, same as the person's name and place the image inside this folder.
2. Place this current folder within "train_images" folder along with the rest of the folders.
3. Now run "Encode_faces_Labview_supported.exe" to create encodings for the newly added images.


***************************** Testing an image ************************************************************************************************

1. To test an image i.e recognize the person in the image, place the image to be tested within "test_image" folder.
2. Now run "Face_Recognition_Labview_supported.exe" to output the findings in "Recognition_results.xml".


***************************** Note ************************************************************************************************************

1. Place the images of interest inside "test_image" folder one at a time. "Face_Recognition_Labview_supported.exe" takes only the first image
   inside the folder as test_image.