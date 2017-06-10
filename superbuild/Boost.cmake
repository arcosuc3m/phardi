#-----------------------------------------------------------------------------
# Boost
#-----------------------------------------------------------------------------


message(STATUS "Installing Boost library.")

set( Boost_Bootstrap_Command )
if( UNIX )
  set( Boost_Bootstrap_Command ./bootstrap.sh )
  set( Boost_b2_Command ./b2 )
else()
  if( WIN32 )
    set( Boost_Bootstrap_Command bootstrap.bat )
    set( Boost_b2_Command b2.exe )
  endif()
endif()

ExternalProject_Add(Boost
  URL "https://dl.bintray.com/boostorg/release/1.64.0/source/boost_1_64_0.tar.bz2"
  BUILD_IN_SOURCE 1
  UPDATE_COMMAND ""
  PREFIX deps/Boost
  PATCH_COMMAND ""
  CONFIGURE_COMMAND ${Boost_Bootstrap_Command}
  BUILD_COMMAND  ${Boost_b2_Command} install
    --without-python
    --without-mpi
    --disable-icu
    --prefix=${CMAKE_BINARY_DIR}/deps/Boost
    --threading=single,multi
    --link=shared
    --variant=release
    -j8
  INSTALL_COMMAND ""
#  INSTALL_COMMAND ${Boost_b2_Command} install 
#    --without-python
#    --without-mpi
#    --disable-icu
#    --prefix=${CMAKE_BINARY_DIR}/INSTALL
#    --threading=single,multi
#    --link=shared
#    --variant=release
#    -j8
)


if( NOT WIN32 )
  set(Boost_LIBRARY_DIR ${CMAKE_BINARY_DIR}/deps/Boost/lib/boost/ )
  set(Boost_INCLUDE_DIR ${CMAKE_BINARY_DIR}/deps/Boost/include/ )
else()
  set(Boost_LIBRARY_DIR ${CMAKE_BINARY_DIR}/deps/Boost/lib/ )
  set(Boost_INCLUDE_DIR ${CMAKE_BINARY_DIR}/deps/Boost/include/boost-1_49/ )
endif()
