ExternalProject_Add(Lapack
  GIT_REPOSITORY "https://github.com/Reference-LAPACK/lapack.git"
  GIT_TAG "v3.7.0"
  PREFIX deps/Lapack
  BUILD_IN_SOURCE
  UPDATE_COMMAND ""
  CMAKE_CACHE_ARGS
    ## CXX should not be needed, but it a cmake default test
    -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
    -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
    -DCBLAS:BOOL=ON
    -DBUILD_SHARED_LIBS:BOOL=ON
  )
