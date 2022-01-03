add_executable(attributes
	attribute.cpp)
	
if (WIN32)	
set_target_properties(attributes PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
endif (WIN32)
	
include_directories(
	../nau/src)

target_link_libraries(attributes nau)

install (TARGETS attributes DESTINATION bin)

