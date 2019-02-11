cmake_minimum_required(VERSION 2.8.12)
set(CMAKE_CXX_STANDARD.11)
add_subdirectory(../dlib dlib_build)


macro(add name)
   add_executable(${name} ${name}.cpp)
   target_link_libraries(${name} dlib::dlib )
   target_link_libraries(${name} opencv_core opencv_highgui opencv_imgproc)
   target_include_directories(${name}  PUBLIC 
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
    third_party/opencv/modules/world/include)
  

endmacro()

add(Face_recognition_SFR)
#add(Encode_faces)


