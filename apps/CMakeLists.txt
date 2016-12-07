ADD_EXECUTABLE(phardi phardi.cpp )
TARGET_LINK_LIBRARIES(phardi 
                      ${Glue} ${VTK_LIBRARIES} ${ITK_LIBRARIES}
                      ${ARMADILLO_LIBRARIES}  
		      ${Boost_LIBRARIES}
		      ${TBB_LIBRARIES}
)

