cmake_minimum_required (VERSION 3.5)

project(NauDemo)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

option(NAU_BUILD_WITH_OPTIX "Build Nau ready for optix" OFF)

set(CMAKE_INSTALL_LIBDIR "${CMAKE_BINARY_DIR}")
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}")

find_package(Threads)
if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-std=c++14 -g)
endif()

if(WIN32) 
	SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /MP" )
	set(CMAKE_DEBUG_POSTFIX "d") 
	# add a user file to auto config the working directory for debugging
	set(VP_WORKING_DIR "${CMAKE_CURRENT_BINARY_DIR}/bin")
	set(VP_COMMAND_ARGS geometry/geometry.xml)
	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/vsconfig.vcxproj.user.in ${CMAKE_CURRENT_BINARY_DIR}/nauGlut/nauGlutDemo.vcxproj.user @ONLY)
	
	set(VP_WORKING_DIR "${CMAKE_CURRENT_BINARY_DIR}/bin")
	set(VP_COMMAND_ARGS "")
	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/vsconfig.vcxproj.user.in ${CMAKE_CURRENT_BINARY_DIR}/composer/composer.vcxproj.user @ONLY)
else()
	find_package(DevIL REQUIRED)
	find_library(AntTweakBar_LIBRARY NAMES
		AntTweakBar
		PATHS
		${PROJECT_SOURCE_DIR}/contrib/AntTweakBar/lib
		NO_DEFAULT_PATH )
	find_library(Lua_LIBRARY NAMES
		lua53
		PATHS
		${PROJECT_SOURCE_DIR}/contrib/lua-5.3.3
		NO_DEFAULT_PATH )
endif(WIN32)

find_package(OpenGL REQUIRED)
include_directories(${OpenGL_INCLUDE_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
add_definitions(${OpenGL_DEFINITIONS})
if(NOT OPENGL_FOUND)
    message(ERROR " OpenGL not found!")
endif(NOT OPENGL_FOUND)


find_package(CUDA)

set(OPTIX_DIR "" CACHE PATH "Path to Optix Install Dir")

if (CUDA_FOUND)
	if (OPTIX_DIR STREQUAL "" )
		if (NOT ${OPTIX_DIR}/include)
		#	message(WARNING "Can't find Otix include folder. Optix option disabled." )
			set (NAU_BUILD_WITH_OPTIX OFF)
		endif(NOT ${OPTIX_DIR}/include)
	endif(OPTIX_DIR STREQUAL "")
endif(CUDA_FOUND)
	
if (CUDA_FOUND AND NOT OPTIX_DIR STREQUAL "")
	option(NAU_BUILD_WITH_OPTIX "Build Nau ready for optix" OFF)
endif(CUDA_FOUND AND NOT OPTIX_DIR STREQUAL "")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

OPTION(INSTALL_PDB "" OFF)

option(OPTION_BUILD_GPU_TESTS "" OFF)
option(OPTION_BUILD_TESTS "" OFF)
#set(BUILD_SHARED_LIBS OFF INTERNAL)
option (BUILD_SHARED_LIBS "" OFF )
add_subdirectory(contrib/glbinding-master)

add_subdirectory(contrib/tinyxml)

set_target_properties (tinyxml PROPERTIES
    FOLDER contrib2/tinyxml)

option(ASSIMP_INSTALL_PDB "" OFF)
option(ASSIMP_BUILD_ASSIMP_TOOLS "" OFF)
option(ASSIMP_BUILD_TESTS "" OFF)
add_subdirectory(contrib/assimp3.3.1)

option(FREEGLUT_BUILD_DEMOS "" OFF)
option(FREEGLUT_BUILD_STATIC_LIBS "" ON)
option(FREEGLUT_BUILD_SHARED_LIBS "" OFF)
option(FREEGLUT_GLES "" OFF)
option(INSTALL_PDB "" OFF)
add_subdirectory(contrib/freeglut-3.0.0)

if (WIN32)
	set_target_properties (tinyxml 
		PROPERTIES FOLDER contrib)
	set_target_properties (freeglut_static 
		PROPERTIES FOLDER contrib)
	set_target_properties (assimp 
		PROPERTIES FOLDER contrib)
	set_target_properties (glbinding 
		PROPERTIES FOLDER contrib)
	set_target_properties (uninstall 
		PROPERTIES FOLDER contrib)
	set_target_properties (zlibstatic 
		PROPERTIES FOLDER contrib)	
endif(WIN32)

add_subdirectory(nau)
add_subdirectory(plugins/passDepthMap)
if (WIN32)
add_subdirectory(plugins/bullet)
add_subdirectory(plugins/physx)
endif(WIN32)

add_subdirectory(nauGLUT)
add_subdirectory(composerImGui)

set_target_properties(
	nauGLUTDemo PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
		RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin
        RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR}/bin
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/bin)
set_target_properties(
	composerImGui PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
		RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin
        RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR}/bin
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/bin)
		
option(WX_COMPOSER "Build ComposerWX" OFF)

if (WX_COMPOSER)
	add_subdirectory(composer)

	set_target_properties(
		composer PROPERTIES
			RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
			RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin
			RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin
			RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR}/bin
			RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/bin)
endif(WX_COMPOSER)