cmake_minimum_required(VERSION 3.2)
project(Panorama)
set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

include_directories ("headers")
include_directories (${OPENCV_INCLUDE_DIR})

add_executable(circle_detection
    circle_detection.cpp
    src/arguments.cpp
)
target_link_libraries (circle_detection ${OpenCV_LIBS})