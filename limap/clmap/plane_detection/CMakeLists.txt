LIMAP_ADD_SOURCES(
    bindings.cc
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")

add_library(clmap_plane_detection STATIC robust_statistics_approach.cc)

target_include_directories(clmap_plane_detection PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../.. ${CMAKE_CURRENT_SOURCE_DIR}/../../../third-party ${CMAKE_CURRENT_SOURCE_DIR}/../../../third-party/PlaneDetection/eigen3)