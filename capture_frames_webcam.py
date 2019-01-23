"""
It is to capture required  number of frames from webcam video.
"""
import cv2
import os

def save_video(video_name):
    """
    It is to save the video captured by webcam.
    :param video_name: pass the name of the video file
    """
    cap = cv2.VideoCapture(-1)
    dims=(640,480)
    fps=20.0
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    out = cv2.VideoWriter(video_name,fourcc,fps, dims)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:

            # write the frame
            out.write(frame)

            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    """
    It is to save n frames from the video captured using webcam as .jpg files.
    :return: returns n frames as .jpg files
    """

    required_frames = 30
    count = 0
    frame_number = 1
    output_filename='output.mp4'
    main_folder="train_frames"

    if not os.path.exists(main_folder):
     os.mkdir(main_folder)
    print('Enter the person name:\t')
    sub_folder = str(input())


    person_folder=main_folder + "/" + sub_folder
    # creating a folder
    if not os.path.exists(person_folder):
        os.mkdir(person_folder)

        save_video(output_filename)
        cap = cv2.VideoCapture(output_filename)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while (cap.isOpened()):

            # Capture frame-by-frame
            ret, frame = cap.read()

            if ret == True:
              # Acquire required number of frames from the video file
              if (count % int(frameCount / required_frames) == 0) and (frame_number < (required_frames + 1)):
                # save frame as JPEG file
                cv2.imwrite(os.path.join(person_folder, sub_folder + "{0}.jpg".format(frame_number)), frame)
                frame_number += 1
              count += 1
            else:
                break
         # When everything done, release the capture
        cap.release()
        os.remove(output_filename)
        cv2.destroyAllWindows()
    else:
          print ('\n' + sub_folder + " already exists")


main()
