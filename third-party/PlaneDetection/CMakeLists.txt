cmake_minimum_required(VERSION 3.17)
project(PlaneDetection)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")

# Macro to add source files to PLANE_DETECTION library.
macro(PLANE_DETECTION_ADD_SOURCES)
    set(SOURCE_FILES "")
    foreach(SOURCE_FILE ${ARGN})
        if(SOURCE_FILE MATCHES "^/.*")
            list(APPEND SOURCE_FILES ${SOURCE_FILE})
        else()
            list(APPEND SOURCE_FILES
                 "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}")
        endif()
    endforeach()
    set(PLANE_DETECTION_SOURCES ${PLANE_DETECTION_SOURCES} ${SOURCE_FILES} PARENT_SCOPE)
endmacro(PLANE_DETECTION_ADD_SOURCES)

add_subdirectory(src)

add_library(plane_detection STATIC ${PLANE_DETECTION_SOURCES})

set(PLANE_DETECTION_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_include_directories(plane_detection PRIVATE ${PLANE_DETECTION_INCLUDE_DIR} ${PROJECT_SOURCE_DIR}/eigen3)
