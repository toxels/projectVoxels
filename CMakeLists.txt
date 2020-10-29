cmake_minimum_required(VERSION 3.17)
project(voxelsWithCuda CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_executable(voxelsWithCuda main.cu rayCast.cu geometry.cuh)
set_target_properties(voxelsWithCuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


include_directories(graphics/include)


find_library(SFML-GRAPHICS sfml-graphics graphics/lib)
find_library(SFML-WINDOW sfml-window graphics/lib)
find_library(SFML-SYSTEM sfml-system graphics/lib)
find_library(SFML-MAIN sfml-main graphics/lib)
find_library(FREETYPE freetype graphics/lib)


target_link_libraries(voxelsWithCuda LINK_PUBLIC ${SFML-GRAPHICS})
target_link_libraries(voxelsWithCuda LINK_PUBLIC ${SFML-WINDOW})
target_link_libraries(voxelsWithCuda LINK_PUBLIC ${SFML-SYSTEM})
target_link_libraries(voxelsWithCuda LINK_PUBLIC ${SFML-MAIN})
target_link_libraries(voxelsWithCuda LINK_PUBLIC ${FREETYPE})