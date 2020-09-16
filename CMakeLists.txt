cmake_minimum_required(VERSION 3.14)
project(projectVoxels)

set(CMAKE_CXX_STANDARD 14)

include_directories(graphics/include)
add_executable(projectVoxels main.cpp geometry.h)

find_library(SFML-GRAPHICS sfml-graphics graphics/lib)
find_library(SFML-WINDOW sfml-window graphics/lib)
find_library(SFML-SYSTEM sfml-system graphics/lib)
find_library(OPENGL32 opengl32 graphics/lib)
find_library(FREETYPE freetype graphics/lib)

target_link_libraries(projectVoxels LINK_PUBLIC ${SFML-GRAPHICS})
target_link_libraries(projectVoxels LINK_PUBLIC ${SFML-WINDOW})
target_link_libraries(projectVoxels LINK_PUBLIC ${SFML-SYSTEM})
target_link_libraries(projectVoxels LINK_PUBLIC ${OPENGL32})
target_link_libraries(projectVoxels LINK_PUBLIC ${FREETYPE})