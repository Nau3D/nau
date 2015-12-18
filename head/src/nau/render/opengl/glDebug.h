#include "nau/config.h"

#ifndef GLDEBUG_H
#define GLDEBUG_H

#include <glbinding/gl/gl.h>
using namespace gl;

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

			/// returns true if th feature is supported
			/// false otherwise
			static bool SetCallback(bool flag);

			/// if -1 keep tracing
			/// if 0 stop tracing
			/// if n > 0 trace frame and return n-1
			static int SetTrace(int numberOfFrames);
			
		protected:

			static bool sInited;
			static bool sCallBackOK;
			static bool sTracing;

			GLDebug(){};
			~GLDebug(){};

			static std::string GetStringForSource(GLenum source);
			static std::string GetStringForType(GLenum type);
			static std::string GetStringForSeverity(GLenum type);

			static void SetTraceCallbacks();

			#if defined(_WIN32) && defined(_DEBUG)	
				static void PrintStack();
			#endif

			static void STDCALL DebugLog(GLenum source,
                       GLenum type,
                       GLuint id,
                       GLenum severity,
                       GLsizei length,
                       const GLchar* message,
					   const void* userParam);

		};
	};
};

#endif