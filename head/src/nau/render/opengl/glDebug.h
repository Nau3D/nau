#include "nau/config.h"

#ifndef GLDEBUG_H
#define GLDEBUG_H

#include <GL/glew.h>

#ifdef _WIN32
#define STDCALL __stdcall
#else
#define STDCALL
#endif

#include <string>

namespace nau {

	namespace render {

		class GLDebug {

		public:
			static bool Init();

		protected:

			static bool Inited;

			GLDebug(){};
			~GLDebug(){};

			static std::string GetStringForSource(GLenum source);
			static std::string GetStringForType(GLenum type);
			static std::string GetStringForSeverity(GLenum type);

			#ifdef _WIN32	
				static void PrintStack();
			#endif

			static void STDCALL DebugLog(GLenum source,
                       GLenum type,
                       GLuint id,
                       GLenum severity,
                       GLsizei length,
                       const GLchar* message,
					   void* userParam);

		};
	};
};

#endif