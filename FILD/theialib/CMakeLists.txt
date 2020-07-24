# cmake needs this line
cmake_minimum_required(VERSION 2.8)

# Define project name
project(runTheia)

# Configure the build to print all details.
SET(CMAKE_VERBOSE_MAKEFILE OFF)

# Configure the build type. This can be passed as parameter too.
#set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)

# Find CUDA, you may need to set CUDA_DIR variable
# to the absolute path to the directory containing CUDAConfig.cmake file
# via the command line or GUI
find_package(CUDA REQUIRED)
# /usr/share/cmake/Modules/FindCUDA.cmake
INCLUDE(FindCUDA)

include_directories(./include /home/tiger/intel/mkl/include/)
# Eigen
# Cmake find modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
set(MIN_EIGEN_VERSION 3.2.0)
find_package(Eigen ${MIN_EIGEN_VERSION} REQUIRED)
if (EIGEN_FOUND)
  message("-- Found Eigen version ${EIGEN_VERSION}: ${EIGEN_INCLUDE_DIRS}")
  include_directories(${EIGEN_INCLUDE_DIRS})
endif (EIGEN_FOUND)


# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the CUDAConfig.cmake file.
# Print some message showing some of them
message(STATUS "CUDA library status:")
message(STATUS "    toolkit root dir: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "    version: ${CUDA_VERSION}")
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
set(CMAKE_PREFIX_PATH "/export/sdb/anshan/software/opencv-3.4/build/install/share/OpenCV")
find_package(OpenCV REQUIRED)

#set(OpenCV_LIBS "opencv_videostab opencv_video opencv_ts opencv_superres opencv_stitching opencv_photo opencv_ocl opencv_objdetect opencv_nonfree opencv_ml opencv_legacy opencv_imgproc opencv_highgui opencv_flann opencv_features2d opencv_core opencv_contrib opencv_calib3d")
#set(OpenCV_INCLUDE_DIRS "/export/anshan/software/opencv-3.4/include")

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

file(GLOB SOURCES "./src/*.cpp")
#set(SOURCES "${PROJECT_SOURCE_DIR}/src/cascade_hasher.cpp;${PROJECT_SOURCE_DIR}/src/theialib.cpp")
#message(STATUS "    include path: ${SOURCES}")


set(Other_LIBS libglog.so)

# Declare the executable target built from your sources
add_executable(runTheia ${SOURCES})
#add_library(theia_rc SHARED  ${SOURCES})

set_target_properties(runTheia PROPERTIES LINKER_LANGUAGE CXX)
#set_target_properties(theia_rc PROPERTIES LINKER_LANGUAGE CXX)

LIST(APPEND CMAKE_CXX_FLAGS "-std=c++11 -w -O3 -ffast-math -Wall")
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_32,code=sm_32)

target_link_libraries(runTheia ${Other_LIBS} ${OpenCV_LIBS})
#target_link_libraries(theia_rc ${Other_LIBS} ${OpenCV_LIBS})

