#ifndef NAU_CONFIG_H
#define NAU_CONFIG_H      
      
// use OpenGL
#define NAU_OPENGL 1
#define GLBINDING_STATIC

// enable and disable functionalities 
// based on OpenGL version

#define NAU_OPENGL_VERSION 450

#define NAU_PLATFORM_WIN32 1

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#define NAU_RENDER_FLAGS

//#define GLINTERCEPTDEBUG
#define NAU_LUA

#define NAU_DEBUG 0

#ifdef NAU_LUA
#pragma comment(lib, "lua53.lib")
#endif

#ifdef _WIN32
#ifdef _DEBUG
#pragma comment(lib, "glbindingd.lib")
#else
#pragma comment(lib, "glbinding.lib")
#endif
#endif

#define GLBINDING_STATIC
#endif // NAU_CONFIG_H
