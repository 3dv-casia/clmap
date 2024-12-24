# code is modified from https://github.com/abner-math/PlaneDetection/blob/master/CommandLine/Makefile

include_directories(../eigen3)


PLANE_DETECTION_ADD_SOURCES(
    connectivitygraph.cpp
    plane.cpp
    circle.cpp
    cylinder.cpp 
    planarpatch.cpp
    planedetector.cpp
)


