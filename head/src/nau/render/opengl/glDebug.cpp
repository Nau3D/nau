#include "nau/render/opengl/glDebug.h"

#include "nau/slogger.h"

#ifdef _WIN32
#include <windows.h>
#include <DbgHelp.h>
#pragma comment(lib,"Dbghelp")
#endif

using namespace nau::render;

bool GLDebug::Inited = false;;

bool 
GLDebug::Init() {

	if (Inited)
		return true;

	// check if the extension is there
	char *s;
	int i = 0, max;
	glGetIntegerv(GL_NUM_EXTENSIONS, &max);
	do {
		s = (char *)glGetStringi(GL_EXTENSIONS, ++i);
	} while (i < max && strcmp(s, "GL_ARB_debug_output") != 0);

	// if we have the extension then ...
	if (s != NULL) {
		// enable sync mode and set the callback
		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB);
		glDebugMessageCallbackARB(DebugLog, NULL);
		return true;
	}
	else {
	// extension has not been loaded
	// report it back
		SLOG("OpenGL Debug Context not enabled\n");
		return false;
	}
}


void /*CALLBACK */
GLDebug::DebugLog(GLenum source,
                       GLenum type,
                       GLuint id,
                       GLenum severity,
                       GLsizei length,
                       const GLchar* message,
					   const void* userParam) {
	
	if (severity == GL_DEBUG_SEVERITY_NOTIFICATION)
		return;

//	SLOG("OpenGL Debug\nType: %s\nSource: %s\nID: %d\nSeverity: %s\n%s",
//		GetStringForType(type).c_str(),
//		GetStringForSource(source).c_str(), id,
//		GetStringForSeverity(severity).c_str(),
//		message);
//#ifdef _WIN32
//	PrintStack();
//#endif
}

// aux function to translate source to string
std::string 
GLDebug::GetStringForSource(GLenum source) {

	switch(source) {
		case GL_DEBUG_SOURCE_API_ARB: 
			return("API");
		case GL_DEBUG_SOURCE_WINDOW_SYSTEM_ARB:
			return("Window System");
		case GL_DEBUG_SOURCE_SHADER_COMPILER_ARB:
			return("Shader Compiler");
		case GL_DEBUG_SOURCE_THIRD_PARTY_ARB:
			return("Third Party");
		case GL_DEBUG_SOURCE_APPLICATION_ARB:
			return("Application");
		case GL_DEBUG_SOURCE_OTHER_ARB:
			return("Other");
		default:
			return("");
	}
}

// aux function to translate severity to string
std::string 
GLDebug::GetStringForSeverity(GLenum severity) {

	switch(severity) {
		case GL_DEBUG_SEVERITY_HIGH_ARB: 
			return("High");
		case GL_DEBUG_SEVERITY_MEDIUM_ARB:
			return("Medium");
		case GL_DEBUG_SEVERITY_LOW_ARB:
			return("Low");
		case GL_DEBUG_SEVERITY_NOTIFICATION:
			return("Notification");
		default:
			return("");
	}
}

// aux function to translate type to string
std::string 
GLDebug::GetStringForType(GLenum type) {

	switch(type) {
		case GL_DEBUG_TYPE_ERROR_ARB: 
			return("Error");
		case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR_ARB:
			return("Deprecated Behaviour");
		case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR_ARB:
			return("Undefined Behaviour");
		case GL_DEBUG_TYPE_PORTABILITY_ARB:
			return("Portability Issue");
		case GL_DEBUG_TYPE_PERFORMANCE_ARB:
			return("Performance Issue");
		case GL_DEBUG_TYPE_OTHER_ARB:
			return("Other");
		default:
			return("");
	}
}

// output the call stack
#ifdef _WIN32
void GLDebug::PrintStack() {

	unsigned int   i;
	void         * stack[ 100 ];
	unsigned short frames;
	SYMBOL_INFO  * symbol;
	HANDLE         process;

	process = GetCurrentProcess();

 	SymSetOptions(SYMOPT_LOAD_LINES);

    SymInitialize( process, NULL, TRUE );

	frames               = CaptureStackBackTrace( 0, 200, stack, NULL );
	symbol               = ( SYMBOL_INFO * )calloc( sizeof( SYMBOL_INFO ) + 256 * sizeof( char ), 1 );
	symbol->MaxNameLen   = 255;
	symbol->SizeOfStruct = sizeof( SYMBOL_INFO );

	for( i = 0; i < frames; i++ )
	{
		SymFromAddr( process, ( DWORD64 )( stack[ i ] ), 0, symbol );
		DWORD  dwDisplacement;
		IMAGEHLP_LINE64 line;

		line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
		if (!strstr(symbol->Name, "GLDebug::") && !strstr(symbol->Name, "wx") &&
			SymGetLineFromAddr64(process, ( DWORD64 )( stack[ i ] ), &dwDisplacement, &line)) {
			
			SLOG("function: %s - line %d", symbol->Name, line.LineNumber);

		}
		//if (0 == strcmp(symbol->Name,"main"))
		//	break;
     }

     free( symbol );
}
#endif