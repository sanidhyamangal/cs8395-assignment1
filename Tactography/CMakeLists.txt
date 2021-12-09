cmake_minimum_required(VERSION 2.8)

project (Tactography)

find_package (ITK REQUIRED)
include (${ITK_USE_FILE})

find_package (VTK REQUIRED)
include (${VTK_USE_FILE})

add_executable (Tactography Tactography.cxx)
add_executable (Tactography_VTK Tactography_VTK.cxx)
add_executable (Tactography_Click Tactography_Click.cxx)


target_link_libraries (Tactography ${ITK_LIBRARIES} ${VTK_LIBRARIES})
target_link_libraries (Tactography_VTK ${ITK_LIBRARIES} ${VTK_LIBRARIES})
target_link_libraries (Tactography_Click ${ITK_LIBRARIES} ${VTK_LIBRARIES})