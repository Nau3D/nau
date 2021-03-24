file(GLOB sources src/*.cpp)
file(GLOB headers src/*.h)

add_library(nauPassGeometry SHARED
	${sources} ${headers})
	
add_dependencies(nauPassGeometry nau)
	
include_directories( 
	../../nau/src
	../../nau/src/nau
)	

if (WIN32)
	add_definitions(-D_WINDLL)
endif(WIN32)
	
target_link_libraries( nauPassGeometry $<TARGET_LINKER_FILE:nau> glbinding ${CUDA_LIBRARIES})
include_directories(${CUDA_INCLUDE_DIRS})


install (TARGETS nauPassGeometry DESTINATION lib)
install(FILES ${headers} DESTINATION include)	

add_custom_command(
	TARGET nauPassGeometry POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
	$<TARGET_FILE:nauPassGeometry>
    ${CMAKE_BINARY_DIR}/bin/nauSettings/plugins/$<CONFIGURATION>/pass/$<TARGET_FILE_NAME:nauPassGeometry>)
					   
if (WIN32)
	set_target_properties (nauPassGeometry 
		PROPERTIES FOLDER plugins)	
endif(WIN32)		
