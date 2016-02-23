# App: test
add_executable(pfiber pfiber.cpp ../src/logging.cpp)
target_link_libraries(pfiber 
                      ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY}
                      ${Boost_LOG_LIBRARY}
                      ${Boost_THREAD_LIBRARY}
                      ${Boost_LOG_SETUP_LIBRARY}
                      ${Glue}  ${VTK_LIBRARIES} ${ITK_LIBRARIES}
                      ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}
                      ${ARMADILLO_LIBRARIES}  
                      ${MKL_LIBRARIES}
)

