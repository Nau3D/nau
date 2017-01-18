file(GLOB sources src/*.cpp)
file(GLOB headers src/*.h)

add_library(nauPassPlugin SHARED
	${sources} ${headers})
	
add_dependencies(nauPassPlugin nau)
	
include_directories( 
	../../nau/src
	../../nau/src/nau
)	

if (WIN32)
	add_definitions(-D_WINDLL)
endif(WIN32)
	
target_link_libraries( nauPassPlugin $<TARGET_LINKER_FILE:nau> glbinding)
	
install (TARGETS nauPassPlugin DESTINATION lib)
install(FILES ${headers} DESTINATION include)	

add_custom_command(
	TARGET nauPassPlugin POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
	$<TARGET_FILE:nauPassPlugin>
    ${CMAKE_BINARY_DIR}/bin/nauSettings/plugins/$<CONFIGURATION>/pass/$<TARGET_FILE_NAME:nauPassPlugin>)
					   
if (WIN32)
	set_target_properties (nauPassPlugin 
		PROPERTIES FOLDER plugins)	
endif(WIN32)		
