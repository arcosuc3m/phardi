ExternalProject_add(OpenBLAS
        GIT_REPOSITORY "git://github.com/xianyi/OpenBLAS.git"
        GIT_TAG "v0.2.8"
        UPDATE_COMMAND ""
        BUILD_IN_SOURCE 1
        PREFIX deps/OpenBLAS
        CONFIGURE_COMMAND ""
        PATCH_COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_LIST_DIR}/../external/Makefile.install.openblas" "<SOURCE_DIR>/Makefile.install"
        BUILD_COMMAND NO_SHARED=1 ${CMAKE_MAKE_PROGRAM} libs netlib
        INSTALL_COMMAND NO_SHARED=1 ${CMAKE_MAKE_PROGRAM} install PREFIX=<INSTALL_DIR>
    )

    set(Openblas_FOUND TRUE)
    #set(Openblas_LIBRARIES ${Openblas_LIBRARY})
    set(BLAS_LAPACK_LIBRARIES ${EXTERNAL_PREFIX}/lib/libopenblas.a)
	if( CMAKE_COMPILER_IS_GNUCC)
		list(APPEND BLAS_LAPACK_LIBRARIES gfortran pthread)
	endif()
    set(BLAS_LAPACK_FOUND ${Openblas_FOUND})
    set(BLAS_LAPACK_INCLUDE_DIRS  ${EXTERNAL_PREFIX}/include)
