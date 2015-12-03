#include "nau.h"

#include "nau/config.h"
#include "nau/slogger.h"
#include "nau/debug/profile.h"
//#include "nau/debug/state.h"
#include "nau/event/eventFactory.h"
#include "nau/loader/cboLoader.h"
#include "nau/loader/iTextureLoader.h"
#include "nau/loader/objLoader.h"
#include "nau/loader/ogreMeshLoader.h"
#include "nau/loader/assimpLoader.h"
#include "nau/loader/patchLoader.h"
#include "nau/loader/projectLoader.h"
//#ifdef GLINTERCEPTDEBUG
//#include "nau/loader/projectLoaderDebugLinker.h"
//#endif //GLINTERCEPTDEBUG
#include "nau/material/uniformBlockManager.h"
#include "nau/render/iAPISupport.h"
#include "nau/render/passFactory.h"
#include "nau/resource/fontManager.h"
#include "nau/scene/sceneFactory.h"
#include "nau/system/file.h"
#include "nau/world/worldFactory.h"

//#include <GL/glew.h>


#ifdef NAU_LUA
extern "C" {
#include<lua/lua.h>
#include <lua/lauxlib.h>
#include <lua/lualib.h>
}
#endif

#include <ctime>

//// added for directory loading
//#ifdef NAU_PLATFORM_WIN32
//#include <dirent.h>
//#else
//#include <dirent.h>
//#include <sys/types.h>
//#endif

using namespace nau;
using namespace nau::system;
using namespace nau::loader;
using namespace nau::scene;
using namespace nau::render;
using namespace nau::resource;
using namespace nau::world;
using namespace nau::material;


static nau::Nau *gInstance = 0;


//bool
//Nau::Init() {
//
//	// UINT
//	Attribs.add(Attribute(FRAME_COUNT, "FRAME_COUNT", Enums::DataType::UINT, true, new unsigned int(0)));
//	// FLOAT
//	Attribs.add(Attribute(TIMER, "TIMER", Enums::DataType::FLOAT, false, new bool(false)));
//
//	NAU->registerAttributes("NAU", &Attribs);
//
//	return true;
//}
//
//
//AttribSet Nau::Attribs;
//bool Nau::Inited = Init();


nau::Nau*
Nau::Create (void) {
	if (0 == gInstance) {
		gInstance = new Nau;
	}	
	return gInstance;
}


nau::Nau*
Nau::GetInstance (void) {

	if (0 == gInstance) {
		Create();
	}

	return gInstance;
}


#ifdef _WINDLL
void 
Nau::SetInstance(Nau * inst) {

	gInstance = inst;
}
#endif


Nau::Nau() :
	m_WindowWidth (0), 
	m_WindowHeight (0), 
//	m_vViewports(),
	m_Inited (false),
	m_Physics (false),
	loadedScenes(0),
	m_ActiveCameraName(""),
	m_Name("Nau"),
	m_RenderFlags(COUNT_RENDER_FLAGS),
	m_UseTangents(false),
	m_UseTriangleIDs(false),
	m_CoreProfile(false),
	isFrameBegin(true), 
	m_DummyVector(),
	m_pRenderManager(NULL),
	m_pMaterialLibManager(NULL),
	m_pResourceManager(NULL),
	m_pEventManager(NULL),
	m_TraceFrames(0),
	m_ProjectName(""),
	m_DefaultState(0),
	m_pAPISupport(0),
	m_pWorld(0), 
	m_LuaState(0),
	m_ProfileResetRequest(false)
{

}


Nau::~Nau() {

	delete MATERIALLIBMANAGER;
	delete EVENTMANAGER;
	delete RENDERMANAGER;
	delete RESOURCEMANAGER;

	delete m_DefaultState; 
	delete m_pAPISupport;

	nau::material::UniformBlockManager::DeleteInstance();
	SLogger::DeleteInstance();

	delete m_pWorld;

	PassFactory::DeleteInstance();

#ifdef NAU_LUA
	if (m_LuaState)
		lua_close(m_LuaState);
#endif
}


bool 
Nau::init (bool context, std::string aConfigFile) {

#if NAU_DEBUG == 1
	CLogger::getInstance().addLog(LEVEL_INFO, "debug.txt");
#endif
	m_AppFolder = File::GetAppFolder();
	//bool result;
	if (true == context) {

		m_pEventManager = new EventManager;
		m_pRenderManager = new RenderManager;
		m_pAPISupport = IAPISupport::GetInstance();
		m_pAPISupport->setAPISupport();
	}	
	m_pResourceManager = new ResourceManager ("."); /***MARK***/ //Get path!!!
	m_pMaterialLibManager = new MaterialLibManager();

	// reset shader block information
	UniformBlockManager::DeleteInstance();

	try {
		ProjectLoader::loadMatLib(m_AppFolder + File::PATH_SEPARATOR + "nauSettings/nauSystem.mlib");
	}
	catch (std::string s) {
		clear();
		throw(s);
	}

	FontManager::addFont("CourierNew10", m_AppFolder + File::PATH_SEPARATOR + "nauSettings/couriernew10.xml", "__FontCourierNew10");

	m_pWorld = WorldFactory::create ("Bullet");

	m_StartTime = (float)clock();// *1000.0 / CLOCKS_PER_MILISEC;
	m_LastFrameTime = NO_TIME;

	m_Inited = true;

	EVENTMANAGER->addListener("WINDOW_SIZE_CHANGED",this);

	m_DefaultState = IState::create();
	m_Viewport = m_pRenderManager->createViewport("defaultFixedVP");

	// Init LUA
#ifdef NAU_LUA
	initLua();
#endif
	PASSFACTORY->loadPlugins();

	return true;
}


const std::string &
Nau::getProjectName() {

	return m_ProjectName;
}


void
Nau::setProjectName(std::string name) {

	m_ProjectName = name;
}


std::string&
Nau::getName() {

	return(m_Name);
}


//float 
//Nau::getPropf(FloatProperty prop) {
//
//	switch (prop) {
//	case TIMER:
//		CLOCKS_PER_MILISEC = CLOCKS_PER_SEC / 1000.0;
//		INV_CLOCKS_PER_MILISEC = 1.0 / CLOCKS_PER_MILISEC;
//		m_FloatProps[TIMER] = clock() * INV_CLOCKS_PER_MILISEC;
//		return m_FloatProps[TIMER];
//
//	default:return(AttributeValues::getPropf(prop));
//	}
//
//}


void
Nau::setProfileResetRequest() {
	m_ProfileResetRequest = true;
}


bool
Nau::getProfileResetRequest() {

	bool aux = m_ProfileResetRequest;
	m_ProfileResetRequest = false;
	return aux;
}

// ----------------------------------------------------
//		Lua Stuff
// ----------------------------------------------------

#ifdef NAU_LUA

void 
luaGetValues(lua_State *l, void *arr, int card, Enums::DataType bdt) {

	float *arrF;
	int *arrI;
	unsigned int *arrUI;

	switch (bdt) {

	case Enums::FLOAT:
		arrF = (float *)arr;
		for (int i = 0; i < card; ++i) {
			lua_pushnumber(l, i + 1); // key
			lua_pushnumber(l, arrF[i]); // value
			lua_settable(l, -3); // two pushes 
		}
		break;
	case Enums::INT:
	case Enums::BOOL:
		arrI = (int *)arr;
		for (int i = 0; i < card; ++i) {
			lua_pushnumber(l, i + 1); // key
			lua_pushnumber(l, arrI[i]); // value
			lua_settable(l, -3); // two pushes 
		}
		break;
	case Enums::UINT:
		arrUI = (unsigned int *)arr;
		for (int i = 0; i < card; ++i) {
			lua_pushnumber(l, i + 1); // key
			lua_pushnumber(l, arrUI[i]); // value
			lua_settable(l, -3); // two pushes 
		}
		break;
	}
}


int
luaGetBuffer(lua_State *l) {

	const char *name = lua_tostring(l, -4);
	size_t offset = (size_t)lua_tointeger(l, -3);
	const char *dataType = lua_tostring(l, -2);

	Enums::DataType dt = Enums::getType(dataType);
	int card = Enums::getCardinality(dt);
	Enums::DataType bdt = Enums::getBasicType(dt);

	IBuffer *buff = RESOURCEMANAGER->getBuffer(name);
	if (buff == NULL) {
		NAU_THROW("Lua getBuffer: invalid buffer: %s", name);
		return 0;
	}

	int size = Enums::getSize(dt);
	void *arr = malloc(size);
	size_t count = buff->getData(offset, size , arr);
	if (size != count) {
		NAU_THROW("Lua getBuffer: buffer %s offset %d, out of bounds", name, (unsigned int)offset);
		return 0;
	}

	luaGetValues(l, arr, card, bdt);

	return 0;
}


int 
luaSetBuffer(lua_State *l) {

	const char *name = lua_tostring(l, -4);
	size_t offset = (size_t)lua_tointeger(l, -3);
	const char *dataType = lua_tostring(l, -2);

	Enums::DataType dt = Enums::getType(dataType);
	int card = Enums::getCardinality(dt);
	Enums::DataType bdt = Enums::getBasicType(dt);
	int size = Enums::getSize(dt);

	IBuffer *buff = RESOURCEMANAGER->getBuffer(name);
	if (buff == NULL) {
		NAU_THROW("Lua getBuffer: invalid buffer: %s", name);
		return 0;
	}

	void *arr;
	float *arrF;
	int *arrI; 
	unsigned int *arrUI;

	switch (bdt) {

	case Enums::FLOAT:
		arrF = (float *)malloc(sizeof(float) * card);
		lua_pushnil(l);
		for (int i = 0; i < card && lua_next(l,-2) != 0; ++i) {
			arrF[i] = (float)lua_tonumber(l, -1);
			lua_pop(l, 1);
		}
		arr = arrF;
		break;
	case Enums::INT:
	case Enums::BOOL:
		arrI = (int *)malloc(sizeof(int) * card);
		lua_pushnil(l);
		for (int i = 0; i < card && lua_next(l, -2) != 0; ++i) {
			arrI[i] = (int)lua_tointeger(l, -1);
			lua_pop(l, 1);
		}
		arr = arrI;
		break;
	case Enums::UINT :
		arrUI = (unsigned int *)malloc(sizeof(unsigned int) * card);
		lua_pushnil(l);
		for (int i = 0; i < card && lua_next(l, -2) != 0; ++i) {
			arrUI[i] = (unsigned int)lua_tointeger(l, -1);
			lua_pop(l, 1);
		}
		arr = arrUI;
		break;
	}

	buff->setSubData(offset, size, arr);

	return 0;
}


int
luaGet(lua_State *l) {

	const char *tipo = lua_tostring(l, -5);
	const char *context = lua_tostring(l, -4);
	const char *component = lua_tostring(l, -3);
	int number = (int)lua_tointeger(l, -2);
	void *arr;
	AttribSet *attr;

	//if (!strcmp(tipo, "CURRENT")) {
	//	attr = NAU->getAttribs(context);
	//	if (attr == NULL)
	//		NAU_THROW("Lua set: Invalid context: %s", context);
	//}
	//else {
		attr = NAU->getAttribs(tipo);
		if (attr == NULL)
			NAU_THROW("Lua get: invalid type: %s", tipo);
	//}

	std::string s = component;
	Enums::DataType dt, bdt;
	int id;
	attr->getPropTypeAndId(s, &dt, &id);
	if (id == -1) {
		NAU_THROW("Lua get: invalid attribute: %s", component);
	}

	int card = Enums::getCardinality(dt);
	bdt = Enums::getBasicType(dt);
	arr = NAU->getAttribute(tipo, context, component, number);
	if (arr == NULL) {
		NAU_THROW("Lua get: Invalid context or number: %s %d", context, number);
	}
	if (!Enums::isBasicType(dt)) {
		arr = ((Data *)(arr))->getPtr();
	}
	luaGetValues(l, arr, card, bdt);

	return 0;
}


int 
luaSet(lua_State *l) {

	const char *tipo = lua_tostring(l, -5);
	const char *context = lua_tostring(l, -4);
	const char *component = lua_tostring(l, -3);
	int number = (int)lua_tointeger(l, - 2);
	Data *arr = NULL;
	AttribSet *attr;

	//if (!strcmp(tipo, "CURRENT")) {
	//	attr = NAU->getAttribs(context);
	//	if (attr == NULL)
	//		NAU_THROW("Lua set: Invalid context: %s", context);
	//}
	//else {
		attr = NAU->getAttribs(tipo);
		if (attr == NULL)
			NAU_THROW("Lua set: invalid type: %s", tipo);
	//}
	std::string s = component;
	Enums::DataType dt, bdt;
	int id;
	attr->getPropTypeAndId(s, &dt, &id);
	if (id == -1)
		NAU_THROW("Lua set: invalid component: %s", component);
	int card = Enums::getCardinality(dt);
	bdt = Enums::getBasicType(dt);
	float *arrF;
	int *arrI; 
	unsigned int *arrUI;

	switch (bdt) {

	case Enums::FLOAT:
		arrF = (float *)malloc(sizeof(float) * card);
		lua_pushnil(l);
		for (int i = 0; i < card && lua_next(l,-2) != 0; ++i) {
			arrF[i] = (float)lua_tonumber(l, -1);
			lua_pop(l, 1);
		}
		switch (dt) {
		case Enums::FLOAT:
			arr = new NauFloat(*arrF); break;
		case Enums::VEC2:
			arr = new vec2(arrF[0], arrF[1]); break;
		case Enums::VEC3:
			arr = new vec3(arrF[0], arrF[1], arrF[2]); break;
		case Enums::VEC4:
			arr = new vec4(arrF[0], arrF[1], arrF[2], arrF[3]); break;
		case Enums::MAT4:
		case Enums::MAT3:
			arr = new mat4(arrF); break;
		default:
			NAU_THROW("Lua set: Type %s not supported", Enums::DataTypeToString[dt].c_str());
		}
		break;
	case Enums::INT:
	case Enums::BOOL:
		arrI = (int *)malloc(sizeof(int) * card);
		lua_pushnil(l);
		for (int i = 0; i < card && lua_next(l, -2) != 0; ++i) {
			arrI[i] = (int)lua_tointeger(l, -1);
			lua_pop(l, 1);
		}
		switch (dt) {
		case Enums::BOOL:
		case Enums::INT:
			arr = new NauInt(arrI[0]); break;
		case Enums::IVEC2:
		case Enums::BVEC2:
			arr = new ivec2(arrI[0], arrI[1]); break;
		case Enums::IVEC3:
		case Enums::BVEC3:
			arr = new ivec3(arrI[0], arrI[1], arrI[2]); break;
		case Enums::IVEC4:
		case Enums::BVEC4:
			arr = new ivec4(arrI[0], arrI[1], arrI[2], arrI[3]); break;
		default:
			NAU_THROW("Lua set: Type %s not supported", Enums::DataTypeToString[dt].c_str());
		}
		break;
	case Enums::UINT :
		arrUI = (unsigned int *)malloc(sizeof(unsigned int) * card);
		lua_pushnil(l);
		for (int i = 0; i < card && lua_next(l, -2) != 0; ++i) {
			arrUI[i] = (unsigned int)lua_tointeger(l, -1);
			lua_pop(l, 1);
		}
		switch (dt) {
		case Enums::UINT:
			arr = new NauUInt(arrUI[0]); break;
		case Enums::UIVEC2:
			arr = new uivec2(arrUI[0], arrUI[1]); break;
		case Enums::UIVEC3:
			arr = new uivec3(arrUI[0], arrUI[1], arrUI[2]); break;
		case Enums::UIVEC4:
			arr = new uivec4(arrUI[0], arrUI[1], arrUI[2], arrUI[3]); break;
		default:
			NAU_THROW("Lua set: Type %s not supported", Enums::DataTypeToString[dt].c_str());
		}
		break;
	default:
		NAU_THROW("Lua set: Type %s not supported", Enums::DataTypeToString[bdt].c_str());
	}

	if (!NAU->setAttribute(tipo, context, component, number, arr))
		NAU_THROW("Lua set: Invalid context: %s", context);

	delete arr;
	return 0;
}


int 
luaSaveTexture(lua_State *l) {

	const char *texName = lua_tostring(l, -1);

	if (!RESOURCEMANAGER->hasTexture(texName))
		NAU_THROW("Lua save texture: invalid texture name");

	nau::material::ITexture *texture = RESOURCEMANAGER->getTexture(texName);

	char s[200];
	sprintf(s,"%s.%d.png", texture->getLabel().c_str(), RENDERER->getPropui(IRenderer::FRAME_COUNT));
	std::string sname = nau::system::File::Validate(s);
	ITextureLoader::Save(texture,ITextureLoader::PNG);

	return 0;
}


int
luaSaveProfile(lua_State *l) {

	const char *fileName = lua_tostring(l, -1);

	std::string prof = Profile::DumpLevels();

	fstream s;
	s.open(fileName, fstream::out);
	s << prof << "\n";
	s.close();

	NAU->setProfileResetRequest();
	//Profile::Reset();
	return 0;
}


void
Nau::initLua() {

	m_LuaState = luaL_newstate();
	luaL_openlibs(m_LuaState);
	lua_pushcfunction(m_LuaState, luaSet);
	lua_setglobal(m_LuaState, "setAttr");
	lua_pushcfunction(m_LuaState, luaGet);
	lua_setglobal(m_LuaState, "getAttr");
	lua_pushcfunction(m_LuaState, luaGetBuffer);
	lua_setglobal(m_LuaState, "getBuffer");
	lua_pushcfunction(m_LuaState, luaSaveTexture);
	lua_setglobal(m_LuaState, "saveTexture");
	lua_pushcfunction(m_LuaState, luaSetBuffer);
	lua_setglobal(m_LuaState, "setBuffer");
	lua_pushcfunction(m_LuaState, luaSaveProfile);
	lua_setglobal(m_LuaState, "saveProfiler");
}


void
Nau::initLuaScript(std::string file, std::string name) {

	int res = luaL_dofile(m_LuaState, file.c_str());
	if (res) {
		NAU_THROW("Error loading lua file: %s", lua_tostring(m_LuaState, -1));
	}
}


void
Nau::callLuaScript(std::string name) {

	lua_getglobal(m_LuaState, name.c_str());
	lua_pcall(m_LuaState, 0, 0, 0);

	// com envio de parâmetros
	//lua_pushnumber(m_LuaState, 12);
	//lua_pushlightuserdata(m_LuaState, &file);
	//lua_pcall(m_LuaState, 2, 0, 0);

	// receber parâmetros
	//lua_pcall(m_LuaState, 0, 1, 0);
	//int k = lua_tonumber(m_LuaState, -1);

}


bool
Nau::callLuaTestScript(std::string name) {

	lua_getglobal(m_LuaState, name.c_str());
	lua_pcall(m_LuaState, 0, 1, 0);
	return (lua_toboolean(m_LuaState, -1) != 0);
}


#endif








// ----------------------------------------------------------
//		GET AND SET ATTRIBUTES
// ----------------------------------------------------------


AttributeValues *
Nau::getCurrentObjectAttributes(std::string &context, int number) {

	IRenderer *renderer = m_pRenderManager->getRenderer();
	IAPISupport *sup = IAPISupport::GetInstance();

	if (context == "CAMERA") {
		return (AttributeValues *)renderer->getCamera().get();
	}
	if (context == "COLOR") {
		return (AttributeValues *)renderer->getMaterial();
	}
	
	if (sup->apiSupport(IAPISupport::IMAGE_TEXTURE) && context == "IMAGE_TEXTURE") {
		return (AttributeValues *)renderer->getImageTexture(number);
	}

	if (context == "LIGHT") {
		return (AttributeValues *)renderer->getLight(number).get();
	}
	if (context == "MATERIAL_TEXTURE") {
		return (AttributeValues *)renderer->getMaterialTexture(number);
	}
	if (context == "PASS") {
		return (AttributeValues *)m_pRenderManager->getCurrentPass();
	}
	if (context == "RENDERER") {
		return (AttributeValues *)renderer;
	}
	if (context == "RENDER_TARGET") {
		return (AttributeValues *)m_pRenderManager->getCurrentPass()->getRenderTarget();
			//m_pResourceManager->getRenderTarget(context);
	}
	if (context == "STATE") {
		return (AttributeValues *)renderer->getState();
	}
	if (context == "TEXTURE") {
		return (AttributeValues *)renderer->getTexture(number);
	}
	if (context == "VIEWPORT") {
		return (AttributeValues *)renderer->getViewport().get();
	}
	// If we get here then we are trying to fetch something that does not exist
	NAU_THROW("Getting an invalid object\ntype: %s", context.c_str());
}


AttributeValues *
Nau::getObjectAttributes(std::string &type, std::string &context, int number) {

	IAPISupport *sup = IAPISupport::GetInstance();
	// From Render Manager
	if (type == "CAMERA") {
		if (m_pRenderManager->hasCamera(context))
			return (AttributeValues *)m_pRenderManager->getCamera(context).get();
	}
	if (type == "LIGHT") {
		if (m_pRenderManager->hasLight(context))
			return (AttributeValues *)m_pRenderManager->getLight(context).get();
	}
	if (type == "SCENE") {
		if (m_pRenderManager->hasScene(context))
			return (AttributeValues *)m_pRenderManager->getScene(context);
	}
	if (type == "PASS") {
		std::string pipName = m_pRenderManager->getActivePipelineName();
		if (m_pRenderManager->hasPass(pipName, context))
			return (AttributeValues *)m_pRenderManager->getPass(pipName, context);
	}
	if (type == "VIEWPORT") {
		if (m_pRenderManager->hasViewport(context))
			return (AttributeValues *)m_pRenderManager->getViewport(context).get();
	}
	if (type == "RENDERER") {
		return (AttributeValues *)RENDERER;
	}

	// From ResourceManager

	if (type == "BUFFER") {
		if (m_pResourceManager->hasBuffer(context))
			return (AttributeValues *)m_pResourceManager->getBuffer(context);
	}
	if (type == "RENDER_TARGET") {
		if (m_pResourceManager->hasRenderTarget(context))
			return (AttributeValues *)m_pResourceManager->getRenderTarget(context);
	}
	if (type == "STATE") {
		if (m_pResourceManager->hasState(context))
			return (AttributeValues *)m_pResourceManager->getState(context);
	}

	// From Materials

	std::string lib, mat;
	std::size_t found = context.find("::");
	if (found != std::string::npos && context.size() > found + 2) {
		lib = context.substr(0, found);
		mat = context.substr(found + 2);
	}

	if (type == "COLOR_MATERIAL") {
		if (m_pMaterialLibManager->hasMaterial(lib, mat))
			return (AttributeValues *)&(m_pMaterialLibManager->getMaterial(lib, mat)->getColor());
	}
	if (type == "TEXTURE") {
		if (m_pMaterialLibManager->hasMaterial(lib, mat))
			return (AttributeValues *)m_pMaterialLibManager->getMaterial(lib, mat)->getTexture(number);
	}
	if (sup->apiSupport(IAPISupport::IMAGE_TEXTURE) && type == "IMAGE_TEXTURE") {
		if (m_pMaterialLibManager->hasMaterial(lib, mat))
			return (AttributeValues *)m_pMaterialLibManager->getMaterial(lib, mat)->getImageTexture(number);
	}
	if (type == "MATERIAL_BUFFER") {
		if (m_pMaterialLibManager->hasMaterial(lib, mat))
			return (AttributeValues *)m_pMaterialLibManager->getMaterial(lib, mat)->getBuffer(number);
	}
	if (type == "TEXTURE_SAMPLER") {
		if (m_pMaterialLibManager->hasMaterial(lib, mat))
			return (AttributeValues *)m_pMaterialLibManager->getMaterial(lib, mat)->getTextureSampler(number);
	}
	if (type == "MATERIAL_TEXTURE") {
		if (m_pMaterialLibManager->hasMaterial(lib, mat))
			return (AttributeValues *)m_pMaterialLibManager->getMaterial(lib, mat)->getMaterialTexture(number);
	}

	// If we get here then we are trying to fetch something that does not exist
	NAU_THROW("Getting an invalid object\ntype: %s\ncontext: %s", 
			type.c_str(), context.c_str());
}


bool
Nau::validateAttribute(std::string type, std::string context, std::string component) {

	int id;
	Enums::DataType dt;
	if (!getObjectAttributes(type, context))
		return false; 
	m_Attributes[type]->getPropTypeAndId(component, &dt, &id); 
	return (id != -1);
}

bool
Nau::validateShaderAttribute(std::string type, std::string context, std::string component) {

	int id;
	Enums::DataType dt;
	std::string what;

	//if (type == "CURRENT")
	//	what = context;
	//else
		what = type;

	if (m_Attributes.count(what) == 0)
		return false;

	m_Attributes[what]->getPropTypeAndId(component, &dt, &id);
	return (id != -1);
}



bool 
Nau::setAttribute(std::string type, std::string context, std::string component, int number, Data *values) {

	int id;
	Enums::DataType dt; 
	AttributeValues *attrVal;

	if (context != "CURRENT") {
//	if (type != "CURRENT") {
		m_Attributes[type]->getPropTypeAndId(component, &dt, &id);
		attrVal = NAU->getObjectAttributes(type, context, number);
	}
	else {
		m_Attributes[type]->getPropTypeAndId(component, &dt, &id);
		attrVal = NAU->getCurrentObjectAttributes(type, number);
//		m_Attributes[context]->getPropTypeAndId(component, &dt, &id);
//		attrVal = NAU->getCurrentObjectAttributes(context, number);
	}

	//attrVal = getObjectAttributes(type, context, number);

	if (attrVal == NULL || id == -1) {
		return false;
	}
	else {
		attrVal->setProp(id, dt, values);
		return true;
	}
}


void *
Nau::getAttribute(std::string type, std::string context, std::string component, int number) {

	int id;
	Enums::DataType dt;
	AttributeValues *attrVal;

	if (context != "CURRENT") {
//	if (type != "CURRENT") {
		attrVal = NAU->getObjectAttributes(type, context, number);
		m_Attributes[type]->getPropTypeAndId(component, &dt, &id);
	}
	else {
		attrVal = NAU->getCurrentObjectAttributes(type, number);
		m_Attributes[type]->getPropTypeAndId(component, &dt, &id);
	}

	if (attrVal == NULL || id == -1) {
		NAU_THROW("Getting an invalid Attribute\ntype: %s\ncontext: %s\ncomponent: %s", 
			type.c_str(), context.c_str(), component.c_str());
	}
	else
		return attrVal->getProp(id, dt);
}


// -----------------------------------------------------------
//		USER ATTRIBUTES
// -----------------------------------------------------------


void 
Nau::registerAttributes(std::string s, AttribSet *attrib) {

	m_Attributes[s] = attrib;
}


bool 
Nau::validateUserAttribContext(std::string context) {

	if (m_Attributes.count(context) != 0)
		return true;
	else
		return false;
}


AttribSet *
Nau::getAttribs(std::string context) {

	if (m_Attributes.count(context) != NULL)
		return m_Attributes[context];
	else
		return NULL;
}


bool 
Nau::validateUserAttribName(std::string context, std::string name) {

	AttribSet *attribs = getAttribs(context);

 // invalid context
	if (attribs == NULL)
		return false;

	std::unique_ptr<Attribute> &a = attribs->get(name);
	if (a->getName() == "NO_ATTR")
		return true;
	else
		return false;
}


void
Nau::deleteUserAttributes() {

	for (auto attr : m_Attributes) {

		attr.second->deleteUserAttributes();
	}
}


std::vector<std::string> &
Nau::getContextList() {

	m_DummyVector.clear();
	for (auto attr : m_Attributes) {
		m_DummyVector.push_back(attr.first);
	}
	return m_DummyVector;
}


// -----------------------------------------------------------
//		EVENTS
// -----------------------------------------------------------


void
Nau::eventReceived(const std::string &sender, const std::string &eventType, 
	const std::shared_ptr<nau::event_::IEventData> &evtData) {

	if (eventType == "WINDOW_SIZE_CHANGED") {
	
		vec3 *evVec = (vec3 *)evtData->getData();
		setWindowSize((unsigned int)evVec->x, (unsigned int)evVec->y);
	}
}


void
Nau::readProjectFile (std::string file, int *width, int *height) {

	try {
		ProjectLoader::load (file, width, height);
		isFrameBegin = true;
	}
	catch (std::string s) {
		clear();
		throw(s);
	}

	setActiveCameraName(RENDERMANAGER->getDefaultCameraName());
}




void
Nau::readDirectory (std::string dirName) {

	std::vector<std::string> files;

	File::RecurseDirectory(dirName, &files);

	if (files.size() == 0) {
		NAU_THROW("Can't open dir %s or directory has no files", dirName.c_str());
	}
	RENDERMANAGER->createScene (dirName);

	for (auto f : files) {
		try {
			NAU->loadAsset (f, dirName);
		}
		catch(std::string &s) {
			throw(s);
		}
	}
//	DIR *dir;
//	struct dirent *ent;
//	bool result = true;
//	char fileName [1024];
//	char sceneName[256];
//
//	clear();
//	sprintf(sceneName,"MainScene"); //,loadedScenes);
//	loadedScenes++;
//	RENDERMANAGER->createScene (dirName);
//	dir = opendir (dirName.c_str());
//
//	if (0 == dir) {
//		NAU_THROW("Can't open dir: %s",dirName);
//	}
//	while (0 != (ent = readdir (dir))) {
//
//#ifdef NAU_PLATFORM_WIN32
//		sprintf (fileName, "%s\\%s", dirName.c_str(), ent->d_name);
//#else
//		sprintf (fileName, "%s/%s", dirName, ent->d_name);						
//#endif
//		try {
//			NAU->loadAsset (fileName, sceneName);
//		}
//		catch(std::string &s) {
//			closedir(dir);
//			throw(s);
//		}
//	}
//	closedir (dir);
	loadFilesAndFoldersAux(dirName, false);	
}


void
Nau::readModel (std::string fileName) throw (std::string) {

	clear();
	bool result = true;
	
	char sceneName[256];

	sprintf(sceneName,"MainScene"); //,loadedScenes);
	loadedScenes++;

	RENDERMANAGER->createScene (sceneName);

	try {
		NAU->loadAsset (fileName, sceneName);
		loadFilesAndFoldersAux(sceneName, false);
	} 
	catch (std::string &s) {
			throw(s);
	}
}


void
Nau::appendModel(std::string fileName) {
	
	char sceneName[256];

	sprintf(sceneName,"MainScene"); //,loadedScenes);
	loadedScenes++;

	RENDERMANAGER->createScene (sceneName);

	try {
		NAU->loadAsset (fileName, sceneName);
		loadFilesAndFoldersAux(sceneName, false);
	}
	catch( std::string &s){
		throw(s);
	}
}


void Nau::loadFilesAndFoldersAux(std::string sceneName, bool unitize) {

	Camera *aNewCam = m_pRenderManager->getCamera ("MainCamera").get();
	std::shared_ptr<Viewport> v = m_pRenderManager->getViewport("defaultFixedVP");//createViewport ("MainViewport", nau::math::vec4(0.0f, 0.0f, 0.0f, 1.0f));
	aNewCam->setViewport (v);

	setActiveCameraName("MainCamera");

	if (unitize) {
		aNewCam->setPerspective (60.0f, 0.01f, 100.0f);
		aNewCam->setPropf4(Camera::POSITION, 0.0f, 0.0f, 5.0f, 1.0f);
	}
	else
		aNewCam->setPerspective (60.0f, 1.0f, 10000.0f);

	// creates a directional light by default
	std::shared_ptr<Light> &l = m_pRenderManager->getLight("MainDirectionalLight");
	l->setPropf4(Light::DIRECTION,1.0f,-1.0f,-1.0f, 0.0f);
	l->setPropf4(Light::COLOR, 0.9f,0.9f,0.9f,1.0f);
	l->setPropf4(Light::AMBIENT,0.5f,0.5f,0.5f,1.0f );

	Pipeline *aPipeline = m_pRenderManager->getPipeline("MainPipeline");
	Pass *aPass = aPipeline->createPass("MainPass");
	aPass->setCamera ("MainCamera");

	aPass->setViewport (v);
	aPass->setPropb(Pass::COLOR_CLEAR, true);
	aPass->setPropb(Pass::DEPTH_CLEAR, true);
	aPass->addLight ("MainDirectionalLight");

	aPass->addScene(sceneName);
	m_pRenderManager->setActivePipeline("MainPipeline");
//	RENDERMANAGER->prepareTriangleIDsAndTangents(true,true);
}


bool
Nau::reload (void) {

	if (false == m_Inited) {
		return false;
	}
	return true;
}


void
Nau::clear() {

	resetFrameCount();
	setActiveCameraName("");
	SceneObject::ResetCounter();
	MATERIALLIBMANAGER->clear();
	EVENTMANAGER->clear();
	EVENTMANAGER->addListener("WINDOW_SIZE_CHANGED", this);
	RENDERMANAGER->clear();
	RESOURCEMANAGER->clear();
	Profile::Reset();
	deleteUserAttributes();

	m_Viewport = RENDERMANAGER->createViewport("defaultFixedVP");

	ProjectLoader::loadMatLib(m_AppFolder + File::PATH_SEPARATOR + "nauSettings/nauSystem.mlib");
}


void 
Nau::setTrace(int frames) {

	m_TraceFrames = frames;
}


bool
Nau::getTraceStatus() {

	return m_TraceOn;
}


void 
Nau::step() {

	IRenderer *renderer = RENDERER;
	float timer = (float)(clock() * INV_CLOCKS_PER_MILISEC);
	if (NO_TIME == m_LastFrameTime) {
		m_LastFrameTime = timer;
	}
	double deltaT = timer - m_LastFrameTime;
	m_LastFrameTime = timer;

	m_TraceOn = m_TraceFrames != 0;
	m_TraceFrames = RENDERER->setTrace(m_TraceFrames);
	if (m_TraceOn) {
		LOG_trace("#NAU(FRAME,START)");
	}

//#ifdef GLINTERCEPTDEBUG
//	addMessageToGLILog("\n#NAU(FRAME,START)");
//#endif //GLINTERCEPTDEBUG

	m_pEventManager->notifyEvent("FRAME_BEGIN", "Nau", "", NULL);


	renderer->resetCounters();
	RESOURCEMANAGER->clearBuffers();

	if (true == m_Physics) {
		m_pWorld->update();
		m_pEventManager->notifyEvent("DYNAMIC_CAMERA", "MainCanvas", "", NULL);
	}

	m_pRenderManager->renderActivePipeline();

	m_pEventManager->notifyEvent("FRAME_END", "Nau", "", NULL);

	unsigned int k = RENDERER->getPropui(IRenderer::FRAME_COUNT);
	if (k == UINT_MAX)
		// 2 avoid issues with run_once and skip_first
		// and allows a future implementation of odd and even frames for
		// ping-pong rendering
		RENDERER->setPropui(IRenderer::FRAME_COUNT, 2);
	else
		RENDERER->setPropui(IRenderer::FRAME_COUNT, ++k);

	//if (getProfileResetRequest())
	//	Profile::Reset();

}


void Nau::stepPass() {

	IRenderer *renderer = RENDERER;
	float timer =(float)(clock() * INV_CLOCKS_PER_MILISEC);
	if (NO_TIME == m_LastFrameTime) {
		m_LastFrameTime = timer;
	}
	double deltaT = timer - m_LastFrameTime;
	m_LastFrameTime = timer;

	Pipeline *p;
	p = RENDERMANAGER->getActivePipeline();
	int lastPass;
	int currentPass;
	lastPass = p->getNumberOfPasses()-1;
	currentPass = p->getPassCounter();

	// if it is the first pass
	if (currentPass == 0) {

		m_pEventManager->notifyEvent("FRAME_BEGIN", "Nau", "", NULL);

		if (m_TraceFrames) {
			LOG_trace("#NAU(FRAME,START)");
		}
//#ifdef GLINTERCEPTDEBUG
//		addMessageToGLILog("\n#NAU(FRAME,START)");
//#endif //GLINTERCEPTDEBUG

		renderer->resetCounters();

		if (true == m_Physics) {
			m_pWorld->update();
			m_pEventManager->notifyEvent("DYNAMIC_CAMERA", "MainCanvas", "", NULL);
		}

	}

	std::string s = RENDERMANAGER->getCurrentPass()->getName();
	if (m_TraceFrames) {
		LOG_trace("\n#NAU(PASS START %s)", s.c_str());
	}
//#ifdef GLINTERCEPTDEBUG
//	addMessageToGLILog(("\n#NAU(PASS,START," + s + ")").c_str());
//#endif //GLINTERCEPTDEBUG

	p->executeNextPass();

	if (m_TraceFrames) {
		LOG_trace("#NAU(PASS END %s)", s.c_str());
	}
//#ifdef GLINTERCEPTDEBUG
//	addMessageToGLILog(("\n#NAU(PASS,END," + s + ")").c_str());
//#endif //GLINTERCEPTDEBUG

	if (currentPass == lastPass) {

		m_pEventManager->notifyEvent("FRAME_END", "Nau", "", NULL);
	
	}

	unsigned int k = RENDERER->getPropui(IRenderer::FRAME_COUNT);
	if (k == ULONG_MAX)
		// 2 avoid issues with run_once and skip_first
		// and allows a future implementation of odd and even frames for
		// ping-pong rendering
		RENDERER->setPropui(IRenderer::FRAME_COUNT, 2);
	else
		RENDERER->setPropui(IRenderer::FRAME_COUNT, ++k);
}


void 
Nau::stepCompleteFrame() {

	Pipeline *p;
	p = RENDERMANAGER->getActivePipeline();
	int totalPasses;
	int currentPass;
	totalPasses = p->getNumberOfPasses();
	currentPass = p->getPassCounter();

	for (int i = currentPass; i < totalPasses; ++i)
		stepPass();
}


void Nau::stepPasses(int n) {

	for (int i = 0; i < n; ++i)
		stepPass();
}


void 
Nau::resetFrameCount() {

	RENDERER->setPropui(IRenderer::FRAME_COUNT, 0);
}


//unsigned long
//Nau::getFrameCount() {
//
//	return RENDERER->FRAME_COUNT];
//
//}


void 
Nau::setActiveCameraName(const std::string &aCamName) {

	if (m_ActiveCameraName != "") {
		EVENTMANAGER->removeListener("CAMERA_MOTION",RENDERMANAGER->getCamera(m_ActiveCameraName).get());
		EVENTMANAGER->removeListener("CAMERA_ORIENTATION",RENDERMANAGER->getCamera(m_ActiveCameraName).get());
	}
	m_ActiveCameraName = aCamName;

	if (m_ActiveCameraName != "") {
		EVENTMANAGER->addListener("CAMERA_MOTION",RENDERMANAGER->getCamera(m_ActiveCameraName).get());
		EVENTMANAGER->addListener("CAMERA_ORIENTATION",RENDERMANAGER->getCamera(m_ActiveCameraName).get());
	}
}


nau::scene::Camera *
Nau::getActiveCamera() {

	if (RENDERMANAGER->hasCamera(m_ActiveCameraName)) {
		return RENDERMANAGER->getCamera(m_ActiveCameraName).get();
	}
	else
		return NULL;
}


float
Nau::getDepthAtCenter() {

	float wz;
	vec2 v = m_Viewport->getPropf2(Viewport::ORIGIN);
	v += m_Viewport->getPropf2(Viewport::SIZE);
	std::shared_ptr<Camera> &cam = RENDERMANAGER->getCamera(m_ActiveCameraName);
	float f = cam->getPropf(Camera::FARP);
	float n = cam->getPropf(Camera::NEARP);
	float p22 = cam->getPropm4(Camera::PROJECTION_MATRIX).at(2,2);
	float p23 = cam->getPropm4(Camera::PROJECTION_MATRIX).at(3,2);
	float p32 = cam->getPropm4(Camera::PROJECTION_MATRIX).at(2, 3);
	float p33 = cam->getPropm4(Camera::PROJECTION_MATRIX).at(3, 3);
	// This must move to glRenderer!!!!
	wz = RENDERER->getDepthAtPoint((int)(v.x*0.5f), (int)(v.y*0.5f));
	wz = (wz  * 2) - 1;
	float pz = (p23 - wz * p33) / (p32*wz - p22);
	SLOG("Depth %f\n",pz); 
	return (pz);
}


void 
Nau::sendKeyToEngine (char keyCode) {

	switch(keyCode) {
	case 'K':
		Profile::Reset();
		break;
	case 'I':
		getDepthAtCenter();
		break;
	}
}


void 
Nau::setClickPosition(int x, int y) {

	RENDERER->setPropi2(IRenderer::MOUSE_CLICK, ivec2(x, y));
}


IWorld&
Nau::getWorld (void) {

	return (*m_pWorld);
}


void
Nau::loadAsset (std::string aFilename, std::string sceneName, std::string params) throw (std::string) {

	File file (aFilename);

	try {
		switch (file.getType()) {
			case File::PATCH:
				PatchLoader::loadScene(RENDERMANAGER->getScene (sceneName), file.getFullPath());
				break;
			case File::COLLADA:
			case File::BLENDER:
			case File::PLY:
			case File::LIGHTWAVE:
			case File::STL:
			case File::TRUESPACE:
				AssimpLoader::loadScene(RENDERMANAGER->getScene (sceneName), file.getFullPath(),params);
				break;
			case File::NAUBINARYOBJECT:
				CBOLoader::loadScene(RENDERMANAGER->getScene(sceneName), file.getFullPath(), params);
				break;
			case File::THREEDS:
				AssimpLoader::loadScene(RENDERMANAGER->getScene (sceneName), file.getFullPath(), params);
				//THREEDSLoader::loadScene (RENDERMANAGER->getScene (sceneName), file.getFullPath(),params);				
				break;
			case File::WAVEFRONTOBJ:
				//AssimpLoader::loadScene(RENDERMANAGER->getScene (sceneName), file.getFullPath(),params);
				OBJLoader::loadScene(RENDERMANAGER->getScene (sceneName), file.getFullPath(), params);				
				break;
			case File::OGREXMLMESH:
				OgreMeshLoader::loadScene(RENDERMANAGER->getScene (sceneName), file.getFullPath());				
				break;
			default:
			  break;
		}
	}
	catch(std::string &s) {
		throw(s);
	}
}


void
Nau::writeAssets (std::string fileType, std::string aFilename, std::string sceneName) {

	if (0 == fileType.compare ("NBO")) {
		CBOLoader::writeScene (RENDERMANAGER->getScene (sceneName), aFilename);
	}
}

void
Nau::setWindowSize (unsigned int width, unsigned int height) {

	m_WindowWidth = width;
	m_WindowHeight = height;
	m_Viewport->setPropf2(Viewport::SIZE, vec2((float)m_WindowWidth, (float)m_WindowHeight));
}


unsigned int 
Nau::getWindowHeight() {

	return(m_WindowHeight);
}


unsigned int 
Nau::getWindowWidth() {

	return(m_WindowWidth);
}


//Viewport*
//Nau::createViewport (const std::string &name, nau::math::vec4 &bgColor) {
//
//	Viewport* v = new Viewport;
//
//	v->setName(name);
//	v->setPropf2 (Viewport::ORIGIN, vec2(0.0f,0.0f));
//	v->setPropf2 (Viewport::SIZE, vec2(m_WindowWidth, m_WindowHeight));
//
//	v->setPropf4(Viewport::CLEAR_COLOR, bgColor);
//	v->setPropb(Viewport::FULL, true);
//
//	m_vViewports[name] = v;
//
//	return v;
//}
//
//
//bool 
//Nau::hasViewport(const std::string &name) {
//
//	return (m_vViewports.count(name) != NULL);
//}
//
//
//Viewport*
//Nau::createViewport (const std::string &name) {
//
//	Viewport* v = new Viewport;
//
//	v->setName(name);
//	v->setPropf2 (Viewport::ORIGIN, vec2(0.0f,0.0f));
////	v->setPropf2 (Viewport::SIZE, vec2(m_WindowWidth, m_WindowHeight));
//	v->setPropb(Viewport::FULL, true);
//
//	m_vViewports[name] = v;
//
//	return v;
//}


//Viewport* 
//Nau::getViewport (const std::string &name) {
//
//	if (m_vViewports.count(name))
//		return m_vViewports[name];
//	else
//		return NULL;
//}


std::shared_ptr<Viewport>
Nau::getDefaultViewport() {
	
	return m_Viewport;
}


//std::vector<std::string> *
//Nau::getViewportNames() {
//
//	std::vector<std::string> *names = new std::vector<std::string>; 
//
//	for( std::map<std::string, nau::render::Viewport*>::iterator iter = m_vViewports.begin(); iter != m_vViewports.end(); ++iter ) {
//      names->push_back(iter->first); 
//    }
//	return names;
//}


void
Nau::enablePhysics(void) {

	m_Physics = true;
}


void
Nau::disablePhysics (void) {

	m_Physics = false;
}


void
Nau::setRenderFlag(RenderFlags aFlag, bool aState) {

	m_RenderFlags[aFlag] = aState;
}


bool
Nau::getRenderFlag(RenderFlags aFlag) {

	return(m_RenderFlags[aFlag]);
}


int 
Nau::picking (int x, int y, std::vector<nau::scene::SceneObject*> &objects, nau::scene::Camera &aCamera) {

	return -1;//	RenderManager->pick (x, y, objects, aCamera);
}

////StateList functions:
//void 
//Nau::loadStateXMLFile(std::string file){
//	State::loadStateXMLFile(file);
//}
//
//
//std::vector<std::string> 
//Nau::getStateEnumNames() {
//	return State::getStateEnumNames();
//}
//
//
//std::string 
//Nau::getState(std::string enumName) {
//	return State::getState(enumName);
//}




RenderManager* 
Nau::getRenderManager (void) {

	return m_pRenderManager;
}


ResourceManager* 
Nau::getResourceManager (void) {

	return m_pResourceManager;
}


MaterialLibManager*
Nau::getMaterialLibManager (void) {

	return m_pMaterialLibManager;
}


EventManager*
Nau::getEventManager (void) {

	return m_pEventManager;
}

nau::render::IRenderer * 
Nau::getRenderer(void)
{
	return getRenderManager()->getRenderer();
}


IAPISupport *
Nau::getAPISupport(void) {

	return m_pAPISupport;
}