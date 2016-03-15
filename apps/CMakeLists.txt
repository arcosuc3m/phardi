# App: phardi
add_executable(phardi phardi.cpp)
target_link_libraries(phardi 
                      ${Glue} ${VTK_LIBRARIES} ${ITK_LIBRARIES}
                      ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}
                      ${ARMADILLO_LIBRARIES}  
                      ${MKL_LIBRARIES}
)

