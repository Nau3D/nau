set(PHYSX_INCLUDE_DIR "" CACHE PATH "Path to PhysX include dir")
set(PHYSX_LIBRARIES_DIR "" CACHE PATH "Path to PhysX lib dir")

file(GLOB sources src/*.cpp)
file(GLOB headers src/*.h)

add_library(nauPhysxPlugin SHARED
	${sources} ${headers})
	
add_dependencies(nauPhysxPlugin nau)
	
include_directories( 
	${PHYSX_INCLUDE_DIR} 
	${PHYSX_INCLUDE_DIR}/../../PxShared/include
	../../nau/src
	../../nau/src/nau
	../../contrib/lua-5.3.3/include
)	
	
target_link_libraries( nauPhysxPlugin
	debug ${PHYSX_LIBRARIES_DIR}/LowLevelDEBUG.lib
	debug ${PHYSX_LIBRARIES_DIR}/LowLevelClothDEBUG.lib
	debug ${PHYSX_LIBRARIES_DIR}/PhysX3CharacterKinematicDEBUG_x64.lib
	debug ${PHYSX_LIBRARIES_DIR}/PhysX3CommonDEBUG_x64.lib
	debug ${PHYSX_LIBRARIES_DIR}/PhysX3CookingDEBUG_x64.lib
	debug ${PHYSX_LIBRARIES_DIR}/PhysX3DEBUG_x64.lib
	debug ${PHYSX_LIBRARIES_DIR}/PhysX3ExtensionsDEBUG.lib
	debug ${PHYSX_LIBRARIES_DIR}/PhysX3VehicleDEBUG.lib
	debug ${PHYSX_LIBRARIES_DIR}/PhysX3GpuDEBUG_x64.lib
	debug ${PHYSX_LIBRARIES_DIR}/SceneQueryDEBUG.lib
	debug ${PHYSX_LIBRARIES_DIR}/SimulationControllerDEBUG.lib
	debug ${PHYSX_INCLUDE_DIR}/../../PxShared/lib/vc14win64/PxFoundationDEBUG_x64.lib
	optimized ${PHYSX_LIBRARIES_DIR}/LowLevel.lib
	optimized ${PHYSX_LIBRARIES_DIR}/LowLevelCloth.lib
	optimized ${PHYSX_LIBRARIES_DIR}/PhysX3CharacterKinematic_x64.lib
	optimized ${PHYSX_LIBRARIES_DIR}/PhysX3Common_x64.lib
	optimized ${PHYSX_LIBRARIES_DIR}/PhysX3Cooking_x64.lib
	optimized ${PHYSX_LIBRARIES_DIR}/PhysX3_x64.lib
	optimized ${PHYSX_LIBRARIES_DIR}/PhysX3Extensions.lib
	optimized ${PHYSX_LIBRARIES_DIR}/PhysX3Vehicle.lib
	optimized ${PHYSX_LIBRARIES_DIR}/PhysX3Gpu_x64.lib
	optimized ${PHYSX_LIBRARIES_DIR}/SceneQuery.lib
	optimized ${PHYSX_LIBRARIES_DIR}/SimulationController.lib
	optimized ${PHYSX_INCLUDE_DIR}/../../PxShared/lib/vc14win64/PxFoundation_x64.lib

	
	$<TARGET_LINKER_FILE:nau>)	
	
target_link_libraries(nauPhysxPlugin "${CMAKE_SOURCE_DIR}/contrib/lua-5.3.3/lua53.lib")
	
#install (TARGETS nauPhysxPlugin DESTINATION lib)
#install(FILES ${headers} DESTINATION include)	

add_custom_command(TARGET nauPhysxPlugin POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
	$<TARGET_FILE:nauPhysxPlugin>
    ${CMAKE_BINARY_DIR}/bin/nauSettings/plugins/$<CONFIG>/physics/$<TARGET_FILE_NAME:nauPhysxPlugin>)
	
if (WIN32)
	set_target_properties (nauPhysxPlugin 
		PROPERTIES FOLDER plugins)	
endif(WIN32)		
	
			   
