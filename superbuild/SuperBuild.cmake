find_package(Git)
if(NOT GIT_FOUND)
  message(ERROR "Cannot find git. git is required for Superbuild")
endif()

option( USE_GIT_PROTOCOL "If behind a firewall turn this off to use http instead." ON)

set(git_protocol "git")

include( ExternalProject )

# Compute -G arg for configuring external projects with the same CMake gener    ator:
if(CMAKE_EXTRA_GENERATOR)
  set(gen "${CMAKE_EXTRA_GENERATOR} - ${CMAKE_GENERATOR}")
else()
  set(gen "${CMAKE_GENERATOR}" )
endif()

set(ep_common_args
    "-DCMAKE_BUILD_TYPE:STRING=Release"
)

FIND_PACKAGE(Armadillo 7.800 QUIET)
IF (ARMADILLO_FOUND)
    MESSAGE("-- FOUND Armadillo. Not installing")
ELSE()
    IF (NOT EXISTS ${CMAKE_BINARY_DIR}/deps/Armadillo)
        MESSAGE("-- NOT FOUND Armadillo. Installing")
        include( ${CMAKE_SOURCE_DIR}/superbuild/Armadillo.cmake )
    ENDIF()
ENDIF()

FIND_PACKAGE(ITK 4.12 QUIET)
IF (ITK_FOUND)
    MESSAGE("-- FOUND ITK. Not Installing")
ELSE()
    IF (NOT EXISTS ${CMAKE_BINARY_DIR}/deps/zlib)
        include( ${CMAKE_SOURCE_DIR}/superbuild/ZLIB.cmake )
    ENDIF()
    IF (NOT EXISTS ${CMAKE_BINARY_DIR}/deps/ITK)
        MESSAGE("-- NOT FOUND ITK. Installing")
        include( ${CMAKE_SOURCE_DIR}/superbuild/ITK.cmake )
    ENDIF()
ENDIF()

FIND_PACKAGE(ArrayFire QUIET)
IF (ArrayFire_FOUND)
    MESSAGE("-- FOUND ArrayFire. Not Installing")
ELSE()

    MESSAGE("-- NOT FOUND ArrayFire. Installing")
    IF (NOT EXISTS ${CMAKE_BINARY_DIR}/deps/Lapack/lib)
        MESSAGE("-- NOT FOUND Lapack. Installing")
        INCLUDE (${CMAKE_SOURCE_DIR}/superbuild/Lapack.cmake )
    ELSE()
        add_custom_target(Lapack SOURCES ${CMAKE_BINARY_DIR}/deps/Lapack/lib)
    ENDIF()

    IF (NOT EXISTS ${CMAKE_BINARY_DIR}/deps/FFTW/lib)
        INCLUDE (${CMAKE_SOURCE_DIR}/superbuild/FFTW.cmake )
    ELSE()
        MESSAGE("-- FOUND deps FFTW. Not installing")
        add_custom_target(FFTW_F SOURCES ${CMAKE_BINARY_DIR}/deps/FFTW)
        add_custom_target(FFTW_D SOURCES ${CMAKE_BINARY_DIR}/deps/FFTW)
        SET (FFTW_ROOT ${CMAKE_BINARY_DIR}/deps/FFTW})
        SET (FFTW_USE_STATIC_LIBS ON)   
        SET (FFTW_INSTALL_DIR ${CMAKE_BINARY_DIR}/deps/FFTW)
    ENDIF()

   # INCLUDE (${CMAKE_SOURCE_DIR}/superbuild/OpenBLAS.cmake )

    INCLUDE(${CMAKE_BINARY_DIR}/deps/Lapack/lib/x86_64-linux-gnu/cmake/lapack-3.7.0/lapack-config.cmake )
    #SET (LAPACK_DIR ${CMAKE_BINARY_DIR}/deps/Lapack/lib)
    #SET (LAPACK_LIBRARIES ${CMAKE_BINARY_DIR}/deps/Lapack/lib)


    IF (NOT EXISTS ${CMAKE_BINARY_DIR}/deps/Boost/lib)
        INCLUDE (${CMAKE_SOURCE_DIR}/superbuild/Boost.cmake )
    ELSE()
        MESSAGE("-- FOUND deps Boost. Not installing")
    ENDIF()

    add_custom_target(Boost SOURCES ${CMAKE_BINARY_DIR}/deps/Boost)
        IF( NOT WIN32 )
            SET (Boost_LIBRARY_DIR ${CMAKE_BINARY_DIR}/deps/Boost/lib/boost/ )
            SET (Boost_INCLUDE_DIR ${CMAKE_BINARY_DIR}/deps/Boost/include/ )
        ELSE()
            SET (Boost_LIBRARY_DIR ${CMAKE_BINARY_DIR}/deps/Boost/lib/ )
            SET (Boost_INCLUDE_DIR ${CMAKE_BINARY_DIR}/deps/Boost/include/boost-1_49/ )
        ENDIF()

    IF (NOT EXISTS ${CMAKE_BINARY_DIR}/deps/ArrayFire/lib)
        include( ${CMAKE_SOURCE_DIR}/superbuild/ArrayFire.cmake )
    ENDIF()
ENDIF()
