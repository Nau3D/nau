#ifndef NAU_CONFIG_H
#define NAU_CONFIG_H      
      
// use OpenGL
#define NAU_OPENGL 1

// enable and disable functionalities 
// based on OpenGL version
<<<<<<< HEAD
#define NAU_OPENGL_VERSION 430
=======
#define NAU_OPENGL_VERSION 330
>>>>>>> origin/debug_wrapper

// use only core features (1 implies no fixed function)
//#define NAU_CORE_OPENGL 1

#define NAU_PLATFORM_WIN32 1

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#define __SLANGER__ 1

#if _MSC_VER >= 1400
#ifndef _CRT_SECURE_NO_DEPRECATE
    #define _CRT_SECURE_NO_DEPRECATE
    #define _CRT_NONSTDC_NO_DEPRECATE
#endif
#endif

#define NAU_RENDER_FLAGS

<<<<<<< HEAD
//#define GLINTERCEPTDEBUG
=======
#define GLINTERCEPTDEBUG
>>>>>>> origin/debug_wrapper

#endif // NAU_CONFIG_H
