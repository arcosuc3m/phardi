#---------------------------------------------------------------------------
# Get and build itk

if( NOT ITK_TAG )
  # 2017-06-01
  set( ITK_TAG "v4.12.0" )
endif()

set( _vtk_args
    -DModule_ITKVtkGlue:BOOL=OFF
    -DModule_ITKLevelSetsv4Visualization:BOOL=OFF
)

ExternalProject_Add(ITK
  GIT_REPOSITORY "${git_protocol}://github.com/InsightSoftwareConsortium/ITK.git"
  GIT_TAG "${ITK_TAG}"
  PREFIX deps/ITK
  UPDATE_COMMAND ""
  CMAKE_GENERATOR ${gen}
  BUILD_IN_SOURCE
  CMAKE_ARGS
    ${ep_common_args}
    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
    -DBUILD_SHARED_LIBS:BOOL=OFF
    -DBUILD_EXAMPLES:BOOL=OFF
    -DBUILD_TESTING:BOOL=OFF
    -DITK_BUILD_DEFAULT_MODULES:BOOL=ON
    -DModule_ITKReview:BOOL=OFF
    -DITK_LEGACY_SILENT:BOOL=OFF
    -DExternalData_OBJECT_STORES:STRING=${ExternalData_OBJECT_STORES}
    "-DITK_USE_SYSTEM_ZLIB:BOOL=ON"
    "-DZLIB_ROOT:PATH=${ZLIB_ROOT}"
    "-DZLIB_INCLUDE_DIR:PATH=${ZLIB_INCLUDE_DIR}"
    "-DZLIB_LIBRARY:FILEPATH=${ZLIB_LIBRARY}"
    ${_vtk_args}
  DEPENDS zlib
  LOG_BUILD 0
)
