ExternalProject_Add(OpenBLAS
  GIT_REPOSITORY "https://github.com/xianyi/OpenBLAS.git"
  GIT_TAG "v0.2.19"
  PREFIX deps/OpenBLAS
  BUILD_IN_SOURCE
  UPDATE_COMMAND ""
  CMAKE_CACHE_ARGS
    ## CXX should not be needed, but it a cmake default test
    -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
    -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
  )
