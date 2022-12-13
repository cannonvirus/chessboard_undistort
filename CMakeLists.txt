cmake_minimum_required(VERSION 3.0.0)
project(findchessboard VERSION 0.2.0)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall")
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(CPACK_PROJECT_NAME ${PROJECT_NAME})
SET(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message(STATUS "CUDA library status:")
    message(STATUS "    version: ${CUDA_VERSION}")
    message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
else ()
    message(FATAL_ERROR "Could not locate CUDA")
endif()

find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
else ()
    message(FATAL_ERROR "Could not locate OpenCV")
endif()

find_package(RapidJSON)
message(STATUS "include path: ${RapidJSON_INCLUDE_DIRS}")

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/opt/nvidia/vpi1/include)
include_directories(/usr/include/eigen3)
# include_directories(/usr/local/include/eigen3)
file(GLOB SOURCE_FILES src/*.cpp)

add_executable(findchessboard main.cpp ${SOURCE_FILES})

target_link_libraries (
    ${CMAKE_PROJECT_NAME}
    ${OpenCV_LIBS}
    /opt/nvidia/vpi1/lib64/libnvvpi.so
    cuda
)