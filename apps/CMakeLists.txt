# App: test
add_executable(pfiber pfiber.cpp)
target_link_libraries(pfiber 
                      ${Glue}  ${VTK_LIBRARIES} ${ITK_LIBRARIES}
                      ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}
                      ${ARMADILLO_LIBRARIES}  
                      ${MKL_LIBRARIES}
)

