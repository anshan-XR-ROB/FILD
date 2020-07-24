cmake_minimum_required(VERSION 2.8)
project(lcd)
include_directories(./include)


include_directories(${GRPC_INCLUDE_DIR})

#OpenCV
set(CMAKE_PREFIX_PATH "/export/sdb/anshan/software/opencv-3.4/build/install/share/OpenCV/")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
list(APPEND LIBS ${OpenCV_LIBS})

#Caffe
set(Caffe_DIR "/export/cheguangfu/workspace/Classification/caffe-augmentation/build/install/")
include_directories(${Caffe_DIR}/include)
find_library(Caffe_LIBRARIES NAMES caffe HINTS ${Caffe_DIR}/lib64)
list(APPEND LIBS ${Caffe_LIBRARIES})

message(STATUS "Caffe_INCLUDE_DIRS: ${Caffe_INCLUDE_DIRS}")
message(STATUS "Caffe_LIBRARIES: ${Caffe_LIBRARIES}")

# Eigen
# Cmake find modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
set(MIN_EIGEN_VERSION 3.2.0)
find_package(Eigen ${MIN_EIGEN_VERSION} REQUIRED)
if (EIGEN_FOUND)
  message("-- Found Eigen version ${EIGEN_VERSION}: ${EIGEN_INCLUDE_DIRS}")
  include_directories(${EIGEN_INCLUDE_DIRS})
endif (EIGEN_FOUND)

#TensorRT + CUDA
#set(CUAD_TensorRT_INCLUDE_DIRS ../include/tensorRT /usr/local/cuda/include/)
#include_directories({TensorRT_INCLUDE_DIRS})
#find_library(NVCAFFE_PARSER nvcaffe_parser ./lib)
#find_library(NVINFER nvinfer ./lib)
#set(CUAD_TensorRT_LIBS ${NVCAFFE_PARSER} ${NVINFER} ${CUDART})

#HNSW
find_library(HNSW hnsw ./lib)
list(APPEND LIBS ${HNSW})

#CUDA
find_package(CUDA) 
include_directories(${CUDA_INCLUDE_DIRS})
set(Other_LIBS libpthread.so libdl.so libboost_system.so libglog.so)
list(APPEND LIBS ${Other_LIBS})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ../lib/)

file(GLOB_RECURSE GEOM_VERIF_SOURCES . src/*.cu)  

# nvcc flags
set(CUDA_NVCC_FLAGS -O3;-G;-g;-std=c++11;-w)
set(CMAKE_BUILD_TYPE "Release")
cuda_add_library(geom_verif SHARED ${GEOM_VERIF_SOURCES})
target_link_libraries(geom_verif ${LIBS})

set(GEOM_VERIF ${CMAKE_CURRENT_SOURCE_DIR}/lib/libgeom_verif.so)
list(APPEND LIBS ${GEOM_VERIF})

set(THEIA_LIB ${CMAKE_CURRENT_SOURCE_DIR}/lib/libtheia_rc.so)
list(APPEND LIBS ${THEIA_LIB})

file(GLOB_RECURSE SOURCES . src/*.cpp)  

set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -msse4.2 -fPIC -o3 -w -fopenmp -mavx2")
set(CMAKE_BUILD_TYPE "Release")

add_executable(lcd ${SOURCES})
target_link_libraries(lcd ${LIBS})
add_dependencies(lcd geom_verif) 

#file(GLOB TOOLS tools/*.cpp)
#foreach(TOOL ${TOOLS})
#  message(STATUS "TARGET:" ${TOOL})
#  get_filename_component(PREFIX ${TOOL} NAME_WE)
#  add_executable(${PREFIX} ${TOOL} ${SOURCES})
#  target_link_libraries(${PREFIX} ${LIBS})
#  add_dependencies(${PREFIX} surf_solver) 
#endforeach(TOOL)

