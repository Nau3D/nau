#ifndef NAU_CONFIG_H
#define NAU_CONFIG_H      
      
// use OpenGL
#define NAU_OPENGL 1

// enable and disable functionalities 
// based on OpenGL version

#define NAU_OPENGL_VERSION 450

#define NAU_PLATFORM_WIN32 1

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#define __SLANGER__ 1

//#if _MSC_VER >= 1400
//#ifndef _CRT_SECURE_NO_DEPRECATE
//    #define _CRT_SECURE_NO_DEPRECATE
//    #define _CRT_NONSTDC_NO_DEPRECATE
//#endif
//#endif

#define NAU_RENDER_FLAGS

#define GLINTERCEPTDEBUG
#define NAU_LUA

#endif // NAU_CONFIG_H
