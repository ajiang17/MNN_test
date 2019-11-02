cmake_minimum_required(VERSION 3.1)
project(MNNTest)

find_package(OpenCV)
message("Opencv Version: ${OPENCV_VERSION}" )

file(GLOB LIBS lib/*.so)

set(INC_DIRS
    ${PROJECT_SOURCE_DIR}/include/mnn
    ${PROJECT_SOURCE_DIR}/include/mnn/include
    ${PROJECT_SOURCE_DIR}/include/mnn/tools
    ${PROJECT_SOURCE_DIR}/include/mnn/tools/cpp
    ${PROJECT_SOURCE_DIR}/include/mnn/source
    ${PROJECT_SOURCE_DIR}/include/mnn/source/backend
    ${PROJECT_SOURCE_DIR}/include/mnn/source/core
    ${PROJECT_SOURCE_DIR}/include/mnn/source/cv
    ${PROJECT_SOURCE_DIR}/include/mnn/source/math
    ${PROJECT_SOURCE_DIR}/include/mnn/source/shape
    ${PROJECT_SOURCE_DIR}/include/mnn/source/src
    ${PROJECT_SOURCE_DIR}/src
)

set(SRC_FILES
    ${PROJECT_SOURCE_DIR}/include/mnn/tools/cpp/revertMNNModel.cpp
    ${PROJECT_SOURCE_DIR}/src/pfld.cpp
    ${PROJECT_SOURCE_DIR}/src/blazeface.cpp
)

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC -std=c++11 -fopenmp")
include_directories(${INC_DIRS})
add_executable(mnist src/mnist.cpp ${SRC_FILES} )
target_link_libraries(mnist ${OpenCV_LIBS} ${LIBS})

add_executable(pfld src/landmark.cpp ${SRC_FILES} )
target_link_libraries(pfld ${OpenCV_LIBS} ${LIBS})

add_executable(blazeface src/detect.cpp ${SRC_FILES} )
target_link_libraries(blazeface ${OpenCV_LIBS} ${LIBS})