# App: test
add_executable(pfiber pfiber.cpp)
target_link_libraries(pfiber 
                      logging
                      multi_intravox_fiber_reconstruction
                      create_kernel_for_rumba 
                      intravox_fiber_reconst_sphdeconv_rumba_sd
                      ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY}
                      ${Boost_LOG_LIBRARY}
                      ${Boost_THREAD_LIBRARY}
                      ${Boost_LOG_SETUP_LIBRARY}
                      ${Glue}  ${VTK_LIBRARIES} ${ITK_LIBRARIES}
                      ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}
                      ${ARMADILLO_LIBRARIES}  
)

