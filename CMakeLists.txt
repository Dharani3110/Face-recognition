cmake_minimum_required(VERSION 2.8.12)
set(CMAKE_CXX_STANDARD.11)
add_subdirectory(../dlib dlib_build)

macro(add name)
   add_executable(${name} ${name}.cpp)
   target_link_libraries(${name} dlib::dlib )
   target_link_libraries(${name} opencv_core opencv_highgui opencv_imgproc opencv_objdetect opencv_videoio)

endmacro()

add(Face_recognition_SFR)
#add(Encode_faces)


