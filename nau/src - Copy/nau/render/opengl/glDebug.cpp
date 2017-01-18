#include "nau/render/opengl/glDebug.h"

#include "nau.h"
#include "nau/slogger.h"
#include "nau/clogger.h"
#include "nau/system/file.h"

#include <glbinding/Binding.h>

#if defined(_WIN32) && defined(_DEBUG)
#include <windows.h>
#include <DbgHelp.h>
#pragma comment(lib,"Dbghelp")
#endif

using namespace nau::render;

bool GLDebug::sInited = false;
bool GLDebug::sCallBackOK = false;
bool GLDebug::sTracing = false;

bool 
GLDebug::Init() {

	if (sInited)
		return true;

	sInited = true;

	// check if the extension is there
	char *s;
	int i = 0, max;
	glGetIntegerv(GL_NUM_EXTENSIONS, &max);
	do {
		s = (char *)glGetStringi(GL_EXTENSIONS, i++);
	} while (i < max && strcmp(s, "GL_ARB_debug_output") != 0);

	// if we have the extension then ...
	if (i < max) {
		// enable sync mode and set the callback
		sCallBackOK = true;
		SetCallback(true);
	}
	else {
	// extension has not been loaded
	// report it back
		SLOG("OpenGL Debug Context not enabled");
		sCallBackOK = false;
	}
	SetTraceCallbacks();
	SetTrace(0);
	//CLogger::GetInstance().addLog(CLogger::LEVEL_TRACE, "nau3Dtrace.txt");
	return sCallBackOK;
}


bool
GLDebug::SetCallback(bool flag) {

	if (!sInited)
		Init();

	if (!sCallBackOK)
		return false;

	if (flag) {
		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB);
		glDebugMessageCallbackARB(DebugLog, NULL);
	}
	else {
		glDebugMessageCallbackARB(NULL, NULL);
	}
	return true;
}


void 
GLDebug::SetTraceCallbacks() {

	glbinding::setUnresolvedCallback([](const glbinding::AbstractFunction & call)
	{
		LOG_trace("UNRESOLVED: %s", call.name());
	});

	// record name of function before calling it
	glbinding::setBeforeCallback([](const glbinding::FunctionCall & call)
	{
		LOG_trace_nr("%s", call.function->name());
	});

	// record parameters and return value after
	glbinding::setAfterCallback([](const glbinding::FunctionCall & call)
	{
		std::string s = "(";

	  for (unsigned i = 0; i < call.parameters.size(); ++i)
	  {
		s += call.parameters[i]->asString();
		if (i < call.parameters.size() - 1)
		  s += ", ";
	  }

	  s += ")";

	  if (call.returnValue)
	  {
		s += " -> " + call.returnValue->asString();
	  }

	  LOG_trace("%s%s", call.function->name() ,s.c_str());
	});

}


int 
GLDebug::SetTrace(int numberOfFrames) {

	if (numberOfFrames == 0) {
		// if we are tracing close the log
		if (sTracing) {
			glbinding::setCallbackMask(glbinding::CallbackMask::None);
			CLogger::CloseLog(CLogger::LEVEL_TRACE);
			EVENTMANAGER->notifyEvent("TRACE_FILE_READY", "", "", NULL);
		}
		sTracing = false;
		return 0;
	}
	else {
		glbinding::setCallbackMask(glbinding::CallbackMask::After | 
							glbinding::CallbackMask::Unresolved |
							glbinding::CallbackMask::ParametersAndReturnValue);

		std::string name = nau::system::File::GetCurrentFolder() + "/__nau3Dtrace";
		nau::system::File::CreateDir(name);
		name += "/Frame_";

		if (NAU->getProjectName() != "")
			name += std::to_string(RENDERER->getPropui(IRenderer::FRAME_COUNT));
		
		name += ".txt";
		CLogger::AddLog(CLogger::LEVEL_TRACE, name);
		sTracing = true;
	}

	if (numberOfFrames > 0)
		return --numberOfFrames;
	else 
		return numberOfFrames;

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

	//LOG_trace(
	SLOG(
		"OpenGL Debug\nType: %s\nSource: %s\nID: %d\nSeverity: %s\n%s",
		GetStringForType(type).c_str(),
		GetStringForSource(source).c_str(), id,
		GetStringForSeverity(severity).c_str(),
		message);
#if defined(_WIN32) && defined(_DEBUG)
	PrintStack();
#endif
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
#if defined(_WIN32) && defined(_DEBUG)
void GLDebug::PrintStack() {

	unsigned int   i;
	void         * stack[ 100 ];
	unsigned short frames;
	SYMBOL_INFO  * symbol;
	HANDLE         process;

	process = GetCurrentProcess();

 	SymSetOptions(SYMOPT_LOAD_LINES);

    SymInitialize( process, NULL, TRUE );

	frames = CaptureStackBackTrace( 0, 200, stack, NULL );
	symbol = ( SYMBOL_INFO * )calloc( sizeof( SYMBOL_INFO ) + 256 * sizeof( char ), 1 );
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
