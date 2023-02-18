add_executable(nauGLUTDemo
	nauGlut.cpp)
	
if (WIN32)	
set_target_properties(nauGLUTDemo PROPERTIES DEBUG_POSTFIX 
${CMAKE_DEBUG_POSTFIX})
endif (WIN32)
	
include_directories(
	../nau/src
	../contrib/lua-5.3.3/include
	../contrib/freeglut-3.0.0/include
	${CMAKE_BINARY_DIR}/contrib/glbinding-master/source/glbinding/include
	../contrib/glbinding-master/source/glbinding/include)	

add_definitions(-DTIXML_USE_STL)

link_directories(${Assimp_BINARY_DIR} ${glbinding_BINARY_DIR} ${freeglut})


target_link_libraries(nauGLUTDemo ${OPENGL_LIBRARIES} )

	
if  (WIN32)
	if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
		file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../contrib/devil/lib64/DevIL.dll 
			DESTINATION ${CMAKE_BINARY_DIR}/bin)
		target_link_libraries(nauGLUTDemo "${CMAKE_SOURCE_DIR}/contrib/devil/lib64/DevIL.lib")
		
	else()
		file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../contrib/devil/lib32/DevIL.dll 
			DESTINATION ${CMAKE_BINARY_DIR}/bin)
		target_link_libraries(nauGLUTDemo "${CMAKE_SOURCE_DIR}/contrib/devil/lib32/DevIL.lib")

	endif( CMAKE_SIZEOF_VOID_P EQUAL 8 )
	target_link_libraries(nauGLUTDemo "${CMAKE_SOURCE_DIR}/contrib/lua-5.3.3/lua53.lib")
	target_link_libraries(nauGLUTDemo nau tinyxml assimp freeglut_static glbinding)
else()
	find_package(GLUT REQUIRED)
	include_directories(${GLUT_INCLUDE_DIRS})
	link_directories(${GLUT_LIBRARY_DIRS})
	target_link_libraries(nauGLUTDemo ${GLUT_LIBRARIES})
	target_link_libraries(nauGLUTDemo nau tinyxml assimp glbinding)

	if (IL_FOUND)
		target_link_libraries(nauGLUTDemo ${IL_LIBRARIES} )
	endif(IL_FOUND)
	#target_link_libraries(nauGLUTDemo "/usr/lib/x86_64-linux-gnu/libdl.so")
	target_link_libraries(nauGLUTDemo "${Lua_LIBRARY}")
endif(WIN32)

install (TARGETS nauGLUTDemo DESTINATION bin)

