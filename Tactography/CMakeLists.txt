cmake_minimum_required(VERSION 2.8)

project (Tactography)

find_package (ITK REQUIRED)
include (${ITK_USE_FILE})

add_executable (Tactography Tactography.cxx)

target_link_libraries (Tactography ${ITK_LIBRARIES})
