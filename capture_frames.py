"""
It is to capture required  number of frames from webcam video.
"""
import cv2
import os

main_folder = "Grabbed_frames"

def capture_video(video_name):
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


def folder_check(sub_folder):

    if not os.path.exists(main_folder):
        os.mkdir(main_folder)
    sub_folder_path = main_folder + "/" + sub_folder

    # creating a new folder
    if not os.path.exists(sub_folder_path):
        folder_status = False
        os.mkdir(sub_folder_path)

    else:
        folder_status = True
        print ('\n' + sub_folder + " already exists")
    return folder_status


def acquire_frames(video_filename,folder_name, required_frames, save_video):

    count = 0
    frame_number = 0
    first_digit = 0
    second_digit = 0

    cap = cv2.VideoCapture(video_filename)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:
            # Acquire required number of frames from the video file
            if (count % int(frameCount / required_frames) == 0) and (frame_number < required_frames):

                # save frame as JPEG file
                cv2.imwrite(os.path.join(main_folder + "/" + folder_name,folder_name + "{0}{1}.jpg".format(first_digit, second_digit)), frame)
                # print(first_digit,second_digit)
                frame_number += 1
                num = frame_number
                second_digit += 1
                if (num % 10 == 0):
                    first_digit += 1
                    second_digit = 0
            count += 1
        else:
            break



        # When everything done, release the capture
    cap.release()
    print("\nFrames grabbed!")
    if not save_video:
        os.remove(output_filename)
    cv2.destroyAllWindows()



def main():
    """
    It is to save n frames from the video captured using webcam as .jpg files.
    :return: returns n frames as .jpg files
    """
    output_filename = 'output.mp4'
    required_frames = 50
    save_video = False
    print('Enter the person name:\t')
    sub_folder = str(input())
    folder_status = folder_check(sub_folder)
    if not folder_status:
       capture_video(output_filename)
       acquire_frames(output_filename, sub_folder, required_frames, save_video)
main()
