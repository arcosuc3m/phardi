ADD_EXECUTABLE(phardi phardi.cpp )
TARGET_LINK_LIBRARIES(phardi 
		      ${CUDA_CUBLAS_LIBRARIES}
                      ${Glue} ${VTK_LIBRARIES} ${ITK_LIBRARIES}
                      ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}
                      ${ARMADILLO_LIBRARIES}  
                      ${MKL_LIBRARIES}
)

