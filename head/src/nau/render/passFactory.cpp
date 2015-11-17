#include "nau/render/passFactory.h"

#include "nau.h"
#include "nau/errors.h"
#include "nau/slogger.h"
#include "nau/render/iAPISupport.h"
#include "nau/render/pass.h"
#include "nau/render/passDepthMap.h"
#include "nau/render/passProfiler.h"

#include "nau/system/file.h"



using namespace nau::render;


PassFactory *PassFactory::Instance = NULL;

PassFactory*
PassFactory::GetInstance (void) {

	if (0 == Instance) {
		Instance = new PassFactory();
	}

	return Instance;
}


void 
PassFactory::DeleteInstance() {

	if (0 != Instance) {

		delete Instance;
		Instance = NULL;
	}
}



#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <exception>
#include <windows.h>

unsigned int 
PassFactory::loadPlugins() {

	std::vector<std::string> files;
	nau::system::File::GetFilesInFolder(".\\nauSettings\\plugins\\", "dll", &files);

	typedef void *(__cdecl *initProc)(void *);
	typedef void *(__cdecl *createPassProc)(const char *);
	typedef char *(__cdecl *getClassNameProc)(void);
	int loaded = 0;

	for (auto fn: files) {
		
		wchar_t wtext[256];
		mbstowcs(wtext, fn.c_str(), fn.size() + 1);//Plus null
		LPWSTR ptr = wtext;
		HINSTANCE mod = LoadLibraryA(fn.c_str());

		if (!mod) {
			SLOG("Library %s wasn't loaded successfully!", fn.c_str());
			break;
		}

		// Get the function and the class exported by the DLL.
		// If you aren't using the MinGW compiler, you may need to adjust
		// this to cope with name mangling (I haven't gone into this here,
		// look it up if you want).
		initProc initFunc = (initProc)GetProcAddress(mod, "init");
		createPassProc createPassFunc = (createPassProc)GetProcAddress(mod, "createPass");
		getClassNameProc getClassNameFunc = (getClassNameProc)GetProcAddress(mod, "getClassName");

		if (!initFunc || !createPassFunc || !getClassNameFunc) {
			SLOG("%s: Invalid Plugin DLL:  'init', 'createPass' and 'getClassName' must be defined", fn.c_str());
			break;
		}
		else
			loaded++;

		initFunc(NAU);
		registerClassFromPlugIn(getClassNameFunc(), createPassFunc);
		// push the objects and modules into our vectors

		SLOG("Plugin %s for pass type %s loaded successfully", fn.c_str(), getClassNameFunc());
	} 

	std::clog << std::endl;

	// Close the file when we are done
	return loaded;
}



PassFactory::PassFactory() {

}


void 
PassFactory::registerClass(const std::string &type, std::shared_ptr<Pass> (*cb)(const std::string &)) {

	m_Creator[type] = cb;
}


void
PassFactory::registerClassFromPlugIn(char *type, void * (*callback)(const char *)) {

	m_PluginCreator[type] = callback;
}


std::shared_ptr<Pass>
PassFactory::create (const std::string &type, const std::string &name) {

	IAPISupport *sup = IAPISupport::GetInstance();
	if (type == "compute" && !sup->apiSupport(IAPISupport::COMPUTE_SHADER))
		NAU_THROW("Compute Shader is not supported");

	if (m_Creator.count(type))
		return (*(std::shared_ptr<Pass>(*)(const std::string &))(m_Creator[type]))(name);
		//return (*(Pass *(*)(const std::string &))(m_Creator[type]))(name);
	else if (m_PluginCreator.count(type)) {
		std::shared_ptr<Pass> *p = (std::shared_ptr<Pass> *)(*(Pass *(*)(const char *))(m_PluginCreator[type]))(name.c_str());
		return *p;
	}

//	if ("default" == type) {
//		return new Pass (name);
//	}
	if ("depthmap2" == type) {
		return std::shared_ptr<Pass>(new PassDepthMap (name));
	}
//	if ("quad" ==  type) {
//		return new PassQuad (name);
//	}
	if ("profiler" == type) {
		return std::shared_ptr<Pass>(new PassProfiler(name));
	}
	return NULL;
//	if ("compute" == type) {
//		return new PassCompute(name);
//	}
//#ifdef NAU_OPTIX
//	if ("optix" == type)
//		return new PassOptix(name);
//#endif
//#ifdef NAU_OPTIX
//#if NAU_OPENGL_VERSION >= 420
//	if ("optixPrime" == type)
//		return new PassOptixPrime(name);
//#endif
//#endif

	//return 0;
}

bool
PassFactory::isClass(const std::string &name) {

	IAPISupport *sup = IAPISupport::GetInstance();
	if (name == "compute" && !sup->apiSupport(IAPISupport::COMPUTE_SHADER))
		return false;

	if (m_Creator.count(name) || m_PluginCreator.count(name))
		return true;
	else
		return false;
}


std::vector<std::string> * 
PassFactory::getClassNames() {

	IAPISupport *sup = IAPISupport::GetInstance();
	std::vector<std::string> *names = new std::vector<std::string>; 

	for (auto s : m_Creator) {
		if (s.first == "compute" || sup->apiSupport(IAPISupport::COMPUTE_SHADER))
			names->push_back(s.first);
	}
	for (auto s : m_PluginCreator) {

		names->push_back(s.first);
	}

	return names;
}
