cmake_minimum_required(VERSION 2.8.12)




#############################################################################
#                                                                           #
#  READ examples/CMakeLists.txt TO SEE HOW TO USE DLIB FROM C++ WITH CMAKE  #
#                                                                           #
#############################################################################



set(CMAKE_CXX_STANDARD.11)
add_subdirectory(../dlib dlib_build)
#add_subdirectory(../opencv third_party)
#add_executable(Face_encodings Face_encodings.cpp)
add_executable(Encode_faces Encode_faces.cpp)
#target_link_libraries(Face_encodings dlib::dlib )
target_link_libraries( Encode_faces dlib::dlib )
#target_link_libraries(Face_encodings opencv_core opencv_highgui opencv_imgproc)
target_link_libraries( Encode_faces opencv_core opencv_highgui opencv_imgproc)

target_include_directories( Encode_faces PUBLIC 
    third_party/opencv/modules/calib3d/include
    third_party/opencv/modules/core/include
    third_party/opencv/modules/cudaarithm/include
    third_party/opencv/modules/cudabgsegm/include
    third_party/opencv/modules/cudacodec/include
    third_party/opencv/modules/cudafeatures2d/include
    third_party/opencv/modules/cudafilters/include
    third_party/opencv/modules/cudaimgproc/include
    third_party/opencv/modules/cudalegacy/include
    third_party/opencv/modules/cudaobjdetect/include
    third_party/opencv/modules/cudaoptflow/include
    third_party/opencv/modules/cudastereo/include
    third_party/opencv/modules/cudawarping/include
    third_party/opencv/modules/cudev/include
    third_party/opencv/modules/dnn/include
    third_party/opencv/modules/features2d/include
    third_party/opencv/modules/flann/include
    third_party/opencv/modules/highgui/include
    third_party/opencv/modules/imgcodecs/include
    third_party/opencv/modules/imgproc/include
    third_party/opencv/modules/ml/include
    third_party/opencv/modules/objdetect/include
    third_party/opencv/modules/photo/include
    third_party/opencv/modules/shape/include
    third_party/opencv/modules/stitching/include
    third_party/opencv/modules/superres/include
    third_party/opencv/modules/ts/include
    third_party/opencv/modules/video/include
    third_party/opencv/modules/videoio/include
    third_party/opencv/modules/videostab/include
    third_party/opencv/modules/viz/include
    third_party/opencv/modules/world/include
)

#if(NOT has_parent)
   # When you call add_subdirectory(dlib) from a parent CMake project dlib's
   # CMake scripts will assume you want to statically compile dlib into
   # whatever you are building rather than create a standalone copy of dlib.
   # This means CMake will build dlib as a static library, disable dlib's
   # install targets so they don't clutter your project, and adjust a few other
   # minor things that are convenient when statically building dlib as part of
   # your own projects.
   #
   # On the other hand, if there is no parent CMake project or if
   # DLIB_IN_PROJECT_BUILD is set to false, CMake will compile dlib as a normal
   # standalone library (either shared or static, based on the state of CMake's
   # BUILD_SHARED_LIBS flag), and include the usual install targets so you can
   # install dlib on your computer via `make install`.  Since the only reason
   # to build this CMakeLists.txt (the one you are reading right now) by itself
   # is if you want to install dlib, we indicate as such by setting
   # DLIB_IN_PROJECT_BUILD to false.
   #set(DLIB_IN_PROJECT_BUILD false)
#endif()

