add_subdirectory(base)
add_subdirectory(ceresbase)
add_subdirectory(solvers)
add_subdirectory(vplib)
add_subdirectory(util)
add_subdirectory(pointsfm)
add_subdirectory(triangulation)
add_subdirectory(merging)
add_subdirectory(undistortion)
add_subdirectory(evaluation)
add_subdirectory(fitting)
add_subdirectory(estimators)
add_subdirectory(optimize)
add_subdirectory(structures)
if(INTERPOLATION_ENABLED)
    add_subdirectory(features)
endif()

# add clmap
add_subdirectory(clmap)

add_subdirectory(_limap)

LIMAP_ADD_STATIC_LIBRARY(limap ${LIMAP_SOURCES})

target_include_directories(limap PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../third-party ${LIMAP_INCLUDE_DIRS})
target_include_directories(_limap PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../third-party ${LIMAP_INCLUDE_DIRS})

target_link_libraries(limap
    ${LIMAP_INTERNAL_LIBRARIES} ${LIMAP_EXTERNAL_LIBRARIES})
target_link_libraries(limap clmap_plane_detection)

