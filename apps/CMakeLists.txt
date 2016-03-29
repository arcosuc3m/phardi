# App: phardi
IF(CUDA_FOUND)
#  CUDA_ADD_LIBRARY(pHARDIGPU ../src/intravox_fiber_reconst_sphdeconv_rumba_sd_kernel.cu)
ENDIF()

# ADD_EXECUTABLE(phardi phardi.cpp ../src/intravox_fiber_reconst_sphdeconv_rumba_sd_gpu.cpp)
ADD_EXECUTABLE(phardi phardi.cpp )
TARGET_LINK_LIBRARIES(phardi 
                      #pHARDIGPU
		      ${CUDA_CUBLAS_LIBRARIES}
                      ${Glue} ${VTK_LIBRARIES} ${ITK_LIBRARIES}
                      ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}
                      ${ARMADILLO_LIBRARIES}  
                      ${MKL_LIBRARIES}
)

