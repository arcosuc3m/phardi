ExternalProject_Add(Armadillo
  GIT_REPOSITORY "https://github.com/conradsnicta/armadillo-code.git"
  GIT_TAG "7.900.x"
  PREFIX deps/Armadillo
  BUILD_IN_SOURCE
  UPDATE_COMMAND ""
  CMAKE_CACHE_ARGS
    ## CXX should not be needed, but it a cmake default test
    -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
    -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
  )
