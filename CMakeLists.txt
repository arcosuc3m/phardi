cmake_minimum_required (VERSION 2.8 FATAL_ERROR)

set(PROJECT_NAME_STR pfiber)
project (${PROJECT_NAME_STR})


FIND_PACKAGE(Armadillo REQUIRED)

FIND_PACKAGE(ITK REQUIRED)
include(${ITK_USE_FILE})

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_STATIC_RUNTIME OFF)
FIND_PACKAGE(Boost 1.55 REQUIRED thread filesystem date_time system log log_setup chrono)

INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIRS} ${ARMADILLO_INCLUDE_DIRS})

# Local project include directory
include_directories("${CMAKE_SOURCE_DIR}/include")

# Build binary directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -DARMA_USE_CXX11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -DARMA_USE_CXX11")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(BLAS REQUIRED)

SET( CMAKE_CXX_FLAGS  " ${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS} ")


#Process subdirectories
add_subdirectory(src)
add_subdirectory(apps)
