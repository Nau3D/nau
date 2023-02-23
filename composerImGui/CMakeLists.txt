file(GLOB imgui ${CMAKE_SOURCE_DIR}/contrib/imgui/*.h ${CMAKE_SOURCE_DIR}/contrib/imgui/*.cpp)	
source_group("imgui" FILES ${imgui})

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

add_executable(composerImGui 
	composerImGui.cpp ${imgui})

if (WIN32)	
set_target_properties(composerImGui PROPERTIES DEBUG_POSTFIX 
${CMAKE_DEBUG_POSTFIX})
endif (WIN32)

include_directories(
	../nau/src
	../contrib/lua-5.3.3/include
	${CMAKE_BINARY_DIR}/contrib/glbinding-master/source/glbinding/include
	../contrib/glbinding-master/source/glbinding/include
	../contrib/glfw
	../contrib/imgui
	${OpenGL_INCLUDE_DIRS}
)	

add_definitions(-DTIXML_USE_STL)
	
target_link_libraries(composerImGui ${OPENGL_LIBRARIES} )
	
	
add_definitions(-DCRT_SECURE_NO_WARNINGS)
add_definitions(-DIMGUI_IMPL_OPENGL_LOADER_GLBINDING)

set(PATH_TO_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../")

if  (WIN32)
	file(COPY "${PROJECT_SOURCE_DIR}/contrib/dirent.h" DESTINATION "${PROJECT_SOURCE_DIR}/contrib/imgui")
	if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
		file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../contrib/devil/lib64/DevIL.dll 
			DESTINATION ${CMAKE_BINARY_DIR}/bin)
		target_link_libraries(composerImGui "${CMAKE_SOURCE_DIR}/contrib/devil/lib64/DevIL.lib")
		
	else()
		file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../contrib/devil/lib32/DevIL.dll 
			DESTINATION ${CMAKE_BINARY_DIR}/bin)
		target_link_libraries(composerImGui "${CMAKE_SOURCE_DIR}/contrib/devil/lib32/DevIL.lib")

	endif( CMAKE_SIZEOF_VOID_P EQUAL 8 )
	target_link_libraries(composerImGui "${CMAKE_SOURCE_DIR}/contrib/lua-5.3.3/lua53.lib")
	target_link_libraries(composerImGui nau tinyxml assimp 	glbinding ${CMAKE_SOURCE_DIR}/contrib/glfw/glfw3.lib)

else()
	list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/contrib/cmake")
	find_package(GLFW3 REQUIRED)
	if (IL_FOUND)
		target_link_libraries(composerImGui ${IL_LIBRARIES} )
	endif(IL_FOUND)
	#target_link_libraries(composerImGui "/usr/lib/x86_64-linux-gnu/libdl.so")
	target_link_libraries(composerImGui nau tinyxml assimp 	glbinding)
	target_link_libraries(composerImGui ${OPENGL_LIBRARIES} )
	target_link_libraries(composerImGui "${Lua_LIBRARY}" ${GLFW3_LIBRARY})
endif(WIN32)


install (TARGETS composerImGui DESTINATION bin)


