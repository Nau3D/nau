#include "nau/render/passFactory.h"

#include "nau.h"
#include "nau/errors.h"
#include "nau/slogger.h"
#include "nau/render/iAPISupport.h"
#include "nau/render/pass.h"
#include "nau/render/passDepthMap.h"
#include "nau/render/passProfiler.h"

#include "nau/system/file.h"

#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <exception>
#ifdef WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif


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




unsigned int 
PassFactory::loadPlugins() {


	std::vector<std::string> files;
	std::string path = nau::system::File::GetAppFolder() + "/nauSettings/plugins/";
#ifdef WIN32	
	path += MY_CONFIG;
	path += "/";
#endif	
	path += "pass/";
#ifdef WIN32
	nau::system::File::GetFilesInFolder(path, "dll", &files);
#else
	nau::system::File::GetFilesInFolder(path, "so", &files);
#endif	
	SLOG("%s", path.c_str());

//#ifdef _DEBUG
//	nau::system::File::GetFilesInFolder(".\\nauSettings\\plugins_d\\pass\\", "dll", &files);
//#else
//	nau::system::File::GetFilesInFolder(".\\nauSettings\\plugins\\pass\\", "dll", &files);
//#endif

#ifdef WIN32
	typedef void *(__cdecl *initProc)(void *);
	typedef void *(__cdecl *createPassProc)(const char *);
	typedef char *(__cdecl *getClassNameProc)(void);
#else
	typedef void *(*initProc)(void *);
	typedef void *(*createPassProc)(const char *);
	typedef void *(*getClassNameProc)(void);
#endif

	int loaded = 0;

	for (auto fn: files) {
		
#ifdef WIN32		
		wchar_t wtext[256];
		mbstowcs(wtext, fn.c_str(), fn.size() + 1);//Plus null
		LPWSTR ptr = wtext;
		HINSTANCE mod = LoadLibraryA(fn.c_str());

		if (!mod) {
			SLOG("Library %s wasn't loaded successfully!", fn.c_str());
			break;
		}

		initProc initFunc = (initProc)GetProcAddress(mod, "init");
		createPassProc createPassFunc = (createPassProc)GetProcAddress(mod, "createPass");
		getClassNameProc getClassNameFunc = (getClassNameProc)GetProcAddress(mod, "getClassName");

#else
		void *lib = dlopen(fn.c_str(), RTLD_NOW);
		if (!lib) {
			SLOG("Library %s wasn't loaded successfully!", fn.c_str());
			break;
		}
		initProc initFunc = (initProc)dlsym(lib, "init");
		createPassProc createPassFunc = (createPassProc)dlsym(lib, "createPass");
		getClassNameProc getClassNameFunc = (getClassNameProc)dlsym(lib, "getClassName");
#endif

		if (!initFunc || !createPassFunc || !getClassNameFunc) {
			SLOG("%s: Invalid Plugin DLL:  'init', 'createPass' and 'getClassName' must be defined", fn.c_str());
			break;
		}
		else
			loaded++;

		initFunc(NAU);
		registerClassFromPlugIn((char *)getClassNameFunc(), createPassFunc);
		// push the objects and modules into our vectors

		SLOG("Plugin %s for pass type %s loaded successfully", fn.c_str(), (char *)getClassNameFunc());
	} 

	std::clog << std::endl;

	// Close the file when we are done
	return loaded;
//#else
//	return 0;
}


PassFactory::PassFactory() {

}


void 
PassFactory::registerClass(const std::string &type, std::shared_ptr<Pass> (*cb)(const std::string &)) {

	m_Creator[type] = cb;
}


void
PassFactory::registerClassFromPlugIn(char *type, void * (*callback)(const char *)) {

	m_PluginCreator[std::string(type)] = (void *)callback;
}


std::shared_ptr<Pass>
PassFactory::create (const std::string &type, const std::string &name) {

	IAPISupport *sup = IAPISupport::GetInstance();
	if (type == "compute" && !sup->apiSupport(IAPISupport::APIFeatureSupport::COMPUTE_SHADER))
		NAU_THROW("Compute Shader is not supported");
	if (type == "mesh" && !sup->apiSupport(IAPISupport::APIFeatureSupport::MESH_SHADER))
		NAU_THROW("Mesh Shader is not supported");

	if (m_Creator.count(type))
		return (*(std::shared_ptr<Pass>(*)(const std::string &))(m_Creator[type]))(name);
		//return (*(Pass *(*)(const std::string &))(m_Creator[type]))(name);
	else if (m_PluginCreator.count(type)) {
		std::shared_ptr<Pass> *p = (std::shared_ptr<Pass> *)(*(Pass *(*)(const char *))(m_PluginCreator[type]))(name.c_str());
		return *p;
	}

	if ("depthmap" == type) {
		return std::shared_ptr<Pass>(new PassDepthMap (name));
	}

	if ("profiler" == type) {
		return std::shared_ptr<Pass>(new PassProfiler(name));
	}
	return NULL;
}

bool
PassFactory::isClass(const std::string &name) {

	IAPISupport *sup = IAPISupport::GetInstance();
	if (name == "compute" && !sup->apiSupport(IAPISupport::APIFeatureSupport::COMPUTE_SHADER))
		return false;
	if (name == "mesh" && !sup->apiSupport(IAPISupport::APIFeatureSupport::MESH_SHADER))
		return false;

	if (m_Creator.count(name) || m_PluginCreator.count(name))
		return true;
	else
		return false;
}

/*
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
*/