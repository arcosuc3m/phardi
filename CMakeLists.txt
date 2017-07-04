CMAKE_MINIMUM_REQUIRED (VERSION 2.8 FATAL_ERROR)

OPTION (USE_SUPERBUILD "Whether or not a superbuild should be invoked" OFF)

if (USE_SUPERBUILD)
  project (SUPERBUILD NONE)
  # execute the superbuild (this script will be invoked again without the
  # USE_SUPERBUILD option this time)
  include (superbuild/SuperBuild.cmake)
  return()                      
endif()

SET(PROJECT_NAME_STR phardi)
PROJECT (${PROJECT_NAME_STR})

SET (${PROJECT_NAME_STR}_VERSION_MAJOR 0)
SET (${PROJECT_NAME_STR}_VERSION_MINOR 3)

SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

# default built type
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: Debug Release."
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

IF(EXISTS ${PROJECT_BINARY_DIR}/deps/Lapack)
    MESSAGE("-- Found Lapack: " ${PROJECT_BINARY_DIR}/deps/Lapack)
    SET(LAPACK_LIBRARIES ${PROJECT_BINARY_DIR}/deps/Lapack/lib/liblapack.so)
ELSE()
    FIND_PACKAGE(LAPACK REQUIRED)
ENDIF()

IF(EXISTS ${PROJECT_BINARY_DIR}/deps/OpenBLAS)
    MESSAGE("-- Found OpenBLAS: " ${PROJECT_BINARY_DIR}/deps/OpenBLAS)
    SET(BLAS_LIBRARIES ${PROJECT_BINARY_DIR}/deps/OpenBLAS/lib/libopenblas.so)
ELSE()
    FIND_PACKAGE(BLAS REQUIRED)
ENDIF()

IF(EXISTS ${PROJECT_BINARY_DIR}/deps/Armadillo)
        SET(ARMADILLO_INCLUDE_DIRS ${PROJECT_BINARY_DIR}/deps/Armadillo/include)
        SET(ARMADILLO_LIBRARY_DIRS ${PROJECT_BINARY_DIR}/deps/Armadillo/lib)
        SET(ARMADILLO_LIBRARIES armadillo)
        INCLUDE_DIRECTORIES(${ARMADILLO_INCLUDE_DIRS})
        LINK_DIRECTORIES(${ARMADILLO_LIBRARY_DIRS})
ELSE()
    FIND_PACKAGE(Armadillo 7.800 QUIET REQUIRED)
    IF (ARMADILLO_FOUND)
        MESSAGE("-- Found Armadillo: " ${ARMADILLO_INCLUDE_DIRS} )
        INCLUDE_DIRECTORIES(${ARMADILLO_INCLUDE_DIRS})
    ENDIF()
ENDIF()

IF(EXISTS ${PROJECT_BINARY_DIR}/deps/ITK)
    INCLUDE(${PROJECT_BINARY_DIR}/deps/ITK/lib/cmake/ITK-4.12/ITKConfig.cmake)
    INCLUDE(${ITK_USE_FILE})
ELSE()
    FIND_PACKAGE(ITK REQUIRED) 
    IF (ITK_FOUND)
         MESSAGE("-- Found ITK: " ${ITK_USE_FILE})
         INCLUDE(${ITK_USE_FILE})
    ENDIF()
ENDIF()


FIND_PACKAGE(CUDA)                                                                                                                                                                                                                                                                                                                          
IF(CUDA_FOUND)                                                                                                                                                        
    FIND_PACKAGE(NVVM REQUIRED) # this FIND script can be found in the ArrayFire CMake example repository                                                                      
    SET(EXTRA_LIBS ${CMAKE_THREAD_LIBS_INIT} ${CUDA_LIBRARIES} ${NVVM_LIB})                                                                                           
ELSE()                                                                                                                                                                
    FIND_PACKAGE(OpenCL)                                                                                                                                              
    SET(EXTRA_LIBS  ${CMAKE_THREAD_LIBS_INIT} ${OpenCL_LIBRARIES})                                                                                                    
ENDIF()   


IF(EXISTS ${PROJECT_BINARY_DIR}/deps/ArrayFire)
    INCLUDE(${PROJECT_BINARY_DIR}/deps/ArrayFire/share/ArrayFire/cmake/ArrayFireConfig.cmake)    
    INCLUDE_DIRECTORIES(${ArrayFire_INCLUDE_DIRS})                                                                                                                        
ELSE()
    FIND_PACKAGE(ArrayFire REQUIRED)                                                                                                                                      
    IF (ArrayFire_FOUND)
        MESSAGE("-- Found ArrayFire: " ${ArrayFire_INCLUDE_DIRS})
        INCLUDE_DIRECTORIES(${ArrayFire_INCLUDE_DIRS})                                                                                                                        
    ENDIF()
ENDIF() 

INCLUDE_DIRECTORIES( "${CMAKE_SOURCE_DIR}/include" "${PROJECT_BINARY_DIR}/include" )

# configure a header file to pass some of the CMake SETtings
# to the source code
CONFIGURE_FILE (
  "${PROJECT_SOURCE_DIR}/include/config.hpp.in"
  "${PROJECT_BINARY_DIR}/include/config.hpp"
  )
 
# Build binary directory
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

INCLUDE(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)

IF(COMPILER_SUPPORTS_CXX11)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
ELSEIF(COMPILER_SUPPORTS_CXX0X)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
ELSE()
    MESSAGE(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a dIFferent C++ compiler.")
ENDIF()

FIND_PACKAGE(OpenMP)
IF (OPENMP_FOUND)
    SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()

SET( CMAKE_CXX_FLAGS  " ${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS} -DARMA_DONT_USE_WRAPPER -DARMA_USE_LAPACK")
SET (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}" )

#Process subdirectories
ADD_SUBDIRECTORY(apps)
