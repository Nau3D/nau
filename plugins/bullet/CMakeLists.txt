
set(BULLET_ROOT "" CACHE PATH "Path to Bullet Install Dir")
find_package(BULLET)
message(STATUS ${BULLET_LIBRARIES})

if (BULLET_FOUND)


file(GLOB sources src/*.cpp)
file(GLOB headers src/*.h)

add_library(nauBulletPlugin SHARED
	${sources})
	
add_dependencies(nauBulletPlugin nau)
	
include_directories( 
	${BULLET_INCLUDE_DIRS} 
	../../nau/src
	../../nau/src/nau
)	
	
target_link_libraries( nauBulletPlugin
	${BULLET_LIBRARIES} 
	$<TARGET_LINKER_FILE:nau>)	
	
target_include_directories(tinyxml PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

install (TARGETS nauBulletPlugin DESTINATION lib)
install(FILES ${headers} DESTINATION include)	

add_custom_command(TARGET nauBulletPlugin POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
	$<TARGET_FILE:nauBulletPlugin>
    ${CMAKE_BINARY_DIR}/bin/nauSettings/plugins/$<CONFIG>/physics/$<TARGET_FILE_NAME:nauBulletPlugin>)
	
if (WIN32)
	set_target_properties (nauBulletPlugin 
		PROPERTIES FOLDER plugins)	
endif(WIN32)		
	
endif(BULLET_FOUND)				   
