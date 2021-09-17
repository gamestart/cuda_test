#  FILENAME:      CMakeLists.txt
#
#  AUTHORS:       Wang Xiaofei    START DATE: Tuesday March 9th 2021
#
#  LAST MODIFIED: Thursday, September 16th 2021, 8:33:48 pm
#
#  CONTACT:       xiaofei.wang@smartmore.com

cmake_minimum_required(VERSION 3.18)

project(cuda-operators LANGUAGES CXX CUDA)

option(CUDAOP_BUILD_TESTING "Whether build test" OFF)

set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/opencv/lib/cmake/opencv4 CACHE STRING "Path to OpenCV")

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 35 37 50 60 70 80)

#for catch2 test framework
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

if (CUDAOP_BUILD_TESTING)
    include(CTest)
    add_subdirectory(tests)
endif()

add_subdirectory(include)
