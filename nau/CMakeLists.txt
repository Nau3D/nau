
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

file(GLOB root src/*.h src/nau/*.h src/nau/*.cpp)	
file(GLOB debug src/nau/debug/*.h src/nau/debug/*.cpp)	
file(GLOB event src/nau/event/*.h src/nau/event/*.cpp)	
file(GLOB geometry src/nau/geometry/*.h src/nau/geometry/*.cpp)	
file(GLOB interface src/nau/interface/*.h src/nau/interface/*.cpp)	
file(GLOB loader src/nau/loader/*.h src/nau/loader/*.cpp)	
file(GLOB material src/nau/material/*.h src/nau/material/*.cpp)	
file(GLOB math src/nau/math/*.h src/nau/math/*.cpp)	
file(GLOB physics src/nau/physics/*.h src/nau/physics/*.cpp)	
file(GLOB render src/nau/render/*.h src/nau/render/*.cpp)	
file(GLOB renderGL src/nau/render/opengl/*.h src/nau/render/opengl/*.cpp)	

if (NAU_BUILD_WITH_OPTIX)
file(GLOB renderOptix src/nau/render/optix/*.h src/nau/render/optix/*.cpp)	
endif()

file(GLOB resource src/nau/resource/*.h src/nau/resource/*.cpp)	
file(GLOB scene src/nau/scene/*.h src/nau/scene/*.cpp)	
file(GLOB system src/nau/system/*.h src/nau/system/*.cpp)	
file(GLOB util src/nau/util/*.h src/nau/util/*.cpp)	
	
source_group("Source Files" FILES ${root})
source_group("Source Files\\debug" FILES ${debug})
source_group("Source Files\\event" FILES ${event})
source_group("Source Files\\geometry" FILES ${geometry})
source_group("Source Files\\interface" FILES ${interface})
source_group("Source Files\\loader" FILES ${loader})
source_group("Source Files\\material" FILES ${material})
source_group("Source Files\\math" FILES ${math})
source_group("Source Files\\physics" FILES ${physics})
source_group("Source Files\\render" FILES ${render})
source_group("Source Files\\render\\gl" FILES ${renderGL})

source_group("Source Files\\resource" FILES ${resource})
source_group("Source Files\\scene" FILES ${scene})
source_group("Source Files\\system" FILES ${system})
source_group("Source Files\\util" FILES ${util})

if (NAU_BUILD_WITH_OPTIX)
	file(GLOB renderRT src/nau/render/rt/*.h src/nau/render/rt/*.cpp)	
	file(GLOB nauSource 
		${root} ${debug} ${event} ${geometry} ${interface} ${loader} ${material} ${math} ${physics} ${render} ${renderGL} ${resource} ${scene} ${system} ${util} ${renderRT})
	source_group("Source Files\\render\\rt" FILES ${renderRT})
	set(NAU_OPTIX 1)
else()
file(GLOB nauSource 
	${root} ${debug} ${event} ${geometry} ${interface} ${loader} ${material} ${math} ${physics} ${render} ${renderGL} ${resource} ${scene} ${system} ${util})
	set(NAU_OPTIX 0)
endif(NAU_BUILD_WITH_OPTIX)

add_library(nau SHARED
	${nauSource} )	

link_directories(${Assimp_BINARY_DIR} ${glbinding_BINARY_DIR} )

target_link_libraries(nau tinyxml assimp glbinding)
target_link_libraries(nau ${OPENGL_LIBRARIES} )

if  (WIN32)
	if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
		file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../contrib/devil/lib64/DevIL.dll 
			DESTINATION ${CMAKE_BINARY_DIR}/bin)
		target_link_libraries(nau "${CMAKE_CURRENT_SOURCE_DIR}/../contrib/devil/lib64/DevIL.lib")
		
		file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../contrib/AntTweakBar/lib/AntTweakBar64.dll 
			DESTINATION ${CMAKE_BINARY_DIR}/bin)
		target_link_libraries(nau "${CMAKE_CURRENT_SOURCE_DIR}/../contrib/AntTweakBar/lib/AntTweakBar64.lib")
		
	else()
		file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../contrib/devil/lib32/DevIL.dll 
			DESTINATION ${CMAKE_BINARY_DIR}/bin)
		target_link_libraries(nau "${CMAKE_CURRENT_SOURCE_DIR}/../contrib/devil/lib32/DevIL.lib")

		file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../contrib/AntTweakBar/lib/AntTweakBar.dll 
			DESTINATION ${CMAKE_BINARY_DIR}/bin)
		target_link_libraries(nau "${CMAKE_CURRENT_SOURCE_DIR}/../contrib/AntTweakBar/lib/AntTweakBar.lib")

	endif( CMAKE_SIZEOF_VOID_P EQUAL 8 )
	target_link_libraries(nau "${CMAKE_CURRENT_SOURCE_DIR}/../contrib/lua-5.3.3/lua53.lib")
else()
	if (IL_FOUND)
		target_link_libraries(nau ${IL_LIBRARIES} )
	endif(IL_FOUND)
	target_link_libraries(nau ${AntTweakBar_LIBRARY})
	target_link_libraries(nau "/usr/lib/x86_64-linux-gnu/libdl.so")
	target_link_libraries(nau "${Lua_LIBRARY}")
endif(WIN32)
	
if (NAU_BUILD_WITH_OPTIX)
	include_directories(${CUDA_INCLUDE_DIRS}
		${OPTIX_DIR}/include)
		target_link_libraries(nau 
		#${OPTIX_DIR}/lib64/optix.7.0.0.lib
		#${OPTIX_DIR}/lib64/optixu.7.0.0.lib
		#${OPTIX_DIR}/lib64/optix_prime.7.0.0.lib
		${CUDA_LIBRARIES})
endif(NAU_BUILD_WITH_OPTIX)	
	
configure_file (
  "${CMAKE_CURRENT_SOURCE_DIR}/src/nau/config.h.in"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/nau/config.h"
  )	

include_directories(
	src
	../contrib/devil
	../contrib/assimp3.3.1/include
	../contrib/tinyxml
	../contrib/lua-5.3.3/include
	../contrib/AntTweakBar/include
	${CMAKE_BINARY_DIR}/contrib/glbinding-master/source/glbinding/include
	../contrib/glbinding-master/source/glbinding/include)	
	
	
if (WIN32)
	include_directories(win)
endif(WIN32)
	
add_definitions(-D_CRT_SECURE_NO_WARNINGS)
add_definitions(-DTIXML_USE_STL -DNOMINMAX)
add_definitions(-DMY_CONFIG="$<CONFIGURATION>")	

install (TARGETS nau DESTINATION lib)


file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/nauSettings DESTINATION ${CMAKE_BINARY_DIR}/bin)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/geometry DESTINATION ${CMAKE_BINARY_DIR}/bin)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/bitmaps DESTINATION ${CMAKE_BINARY_DIR}/bin)

set_target_properties(
	nau PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
		RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin
        RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR}/bin
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/bin)
