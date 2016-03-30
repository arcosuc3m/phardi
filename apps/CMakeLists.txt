ADD_EXECUTABLE(phardi phardi.cpp )
TARGET_LINK_LIBRARIES(phardi 
		      ${ArrayFire_LIBRARIES} 
		      ${EXTRA_LIBS}
                      ${Glue} ${VTK_LIBRARIES} ${ITK_LIBRARIES}
                      ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}
                      ${ARMADILLO_LIBRARIES}  
                      ${MKL_LIBRARIES}
)

