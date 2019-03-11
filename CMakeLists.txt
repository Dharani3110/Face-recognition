

cmake_minimum_required(VERSION 2.8.12)
set(CMAKE_CXX_STANDARD.11)
find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)

macro(add name)
  add_executable(${name} ${name}.cpp)
  TARGET_LINK_LIBRARIES(${name} ${dlib_LIBRARIES})
  INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIR})
  INCLUDE_DIRECTORIES(${dlib_INCLUDE_DIR})
  TARGET_LINK_LIBRARIES(${name} ${OpenCV_LIBRARIES})


endmacro()

add(Nuera_Face_recognition)
