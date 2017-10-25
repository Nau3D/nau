
file(GLOB root src/*.h src/*.h src/*.cpp)	
file(GLOB dialogs src/dialogs/*.h src/dialogs/*.cpp)	

source_group("Source Files" FILES ${root})
source_group("Source Files\\dialogs" FILES ${dialogs})

add_executable(composer 
	${root} ${dialogs} )	

set_target_properties(composer PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})

add_dependencies(composer nau)

find_package(wxWidgets COMPONENTS adv core base gl propgrid)

include_directories(
	src
	../nau/src
	../nau/src/nau
	${wxWidgets_INCLUDE_DIRS}
	${wxWidgets_LIB_DIR}/${wxWidgets_CONFIGURATION})
	
if (WIN32)
	include_directories(../nau/win)
	set_target_properties(composer PROPERTIES LINK_FLAGS "/SUBSYSTEM:WINDOWS")
else()	
	add_definitions(`wx-config --cxxflags` `wx-config --libs`)
endif(WIN32)

add_definitions(-DWXUSINGDLL)

install (TARGETS composer DESTINATION ${CMAKE_BINARY_DIR})	

target_link_libraries(composer $<TARGET_LINKER_FILE:nau> glbinding)
target_link_libraries(composer ${wxWidgets_LIBRARIES})
target_link_libraries(composer ${OPENGL_LIBRARIES} )

