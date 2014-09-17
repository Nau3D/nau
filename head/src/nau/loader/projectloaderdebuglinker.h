#include <string>

/*
This header file is for the debug linker, it links with
GL Intercept's debug library allowing direct configuration
changes.
*/

class GLIEnums
{

public:

	typedef enum FunctionType {
					INT,
					UINT,
					BOOL,
					STRING,
					UINTARRAY,
					STRINGARRAY,
					NIL
	};

};

void initGLInterceptFunctions();

void useGLIFunction(void *functionSetPointer, void *value);
void useGLIClearFunction(void *functionSetPointer);
void *getGLIFunction(const char *name, const char *mapName);
unsigned int getGLIFunctionType(void *functionSetPointer);

void startGLIConfiguration();

void addPlugin(const char *pluginName, const char *pluginDLLName, const char *pluginConfigData);
void clearPlugins();
void startGlilog();