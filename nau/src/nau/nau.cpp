#include "nau.h"

#include "nau/config.h"
#include "nau/slogger.h"
#include "nau/debug/profile.h"
#include "nau/geometry/sphere.h"
#include "nau/interface/interface.h"
#include "nau/event/eventFactory.h"
#include "nau/loader/bufferLoader.h"
#include "nau/loader/cboLoader.h"
#include "nau/loader/iTextureLoader.h"
#include "nau/loader/objLoader.h"
#include "nau/loader/ogreMeshLoader.h"
#include "nau/loader/assimpLoader.h"
#include "nau/loader/patchLoader.h"
#include "nau/loader/projectLoader.h"
#include "nau/material/uniformBlockManager.h"
#include "nau/render/iAPISupport.h"
#include "nau/render/passFactory.h"
#include "nau/render/passProcessTexture.h"
#include "nau/render/passProcessBuffer.h"
#include "nau/resource/fontManager.h"
#include "nau/scene/sceneFactory.h"
#include "nau/system/file.h"


#if NAU_LUA == 1

extern "C" {
#include <lua/lua.h>
#include <lua/lauxlib.h>
#include <lua/lualib.h>
}

std::map<std::string, std::string> Nau::LuaScriptNames;
std::set<string> Nau::LuaFilesWithIssues;
std::string Nau::LuaCurrentScript;

#endif

#include <algorithm>
#include <ctime>
#include <typeinfo>

using namespace nau;
using namespace nau::geometry;
using namespace nau::loader;
using namespace nau::material;
using namespace nau::render;
using namespace nau::resource;
using namespace nau::scene;
using namespace nau::system;


nau::Nau *Nau::Instance = NULL;

#if NAU_LUA == 1
lua_State *Nau::LuaState = NULL;
#endif

nau::INau*
Nau::Create (void) {
	if (0 == Instance) {
		Instance = new Nau;
	}	
	return Instance;
}


nau::INau*
Nau::GetInstance (void) {

	if (0 == Instance) {
		Create();
	}

	return Instance;
}


void 
Nau::SetInstance(Nau * inst) {

	Instance = inst;
}


Nau::Nau() :
	m_WindowWidth (0), 
	m_WindowHeight (0), 
	m_Inited (false),
	m_Physics (false),
	loadedScenes(0),
	m_ActiveCameraName(""),
	m_Name("Nau"),
	m_RenderFlags(COUNT_RENDER_FLAGS),
	m_UseTangents(false),
	m_UseTriangleIDs(false),
	m_CoreProfile(false),
	m_pRenderManager(NULL),
	m_pMaterialLibManager(NULL),
	m_pResourceManager(NULL),
	m_pEventManager(NULL),
	m_TraceFrames(0),
	m_ProjectName(""),
	m_DefaultState(0),
	m_pAPISupport(0),
	m_ProjectFolder(""),
	m_ProfileResetRequest(false)
{
}


Nau::~Nau() {

	delete MATERIALLIBMANAGER;
	delete RENDERMANAGER;
	delete RESOURCEMANAGER;
	m_Viewport.reset();
	delete EVENTMANAGER;
	m_pEventManager = NULL;
	delete m_pPhysicsManager;

	delete m_DefaultState; 
	delete m_pAPISupport;

	nau::material::UniformBlockManager::DeleteInstance();
	SLogger::DeleteInstance();

	PassFactory::DeleteInstance();

#if NAU_LUA == 1
	if (LuaState)
		lua_close(LuaState);
#endif

	Instance = NULL;
	delete INTERFACE_MANAGER;
}


bool 
Nau::init (bool trace) {

#if NAU_DEBUG == 1
	CLogger::getInstance().addLog(LEVEL_INFO, "debug.txt");
#endif
	m_AppFolder = File::GetAppFolder();
	m_pEventManager = EventManager::GetInstance();
	m_pRenderManager = new RenderManager;

	if (trace) {
		RENDERER->setTrace(-1);
		m_TraceFrames = -1;
	}
	m_pAPISupport = IAPISupport::GetInstance();
	m_pAPISupport->setAPISupport();
	m_pPhysicsManager = nau::physics::PhysicsManager::GetInstance();	
	m_pResourceManager = new ResourceManager ("."); 
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

	m_StartTime = (float)clock();// *1000.0 / CLOCKS_PER_MILISEC;
	m_LastFrameTime = NO_TIME;

	m_Inited = true;

	EVENTMANAGER->addListener("WINDOW_SIZE_CHANGED",this);

	m_DefaultState = IState::create();
	m_Viewport = m_pRenderManager->createViewport("__nauDefault");
	m_Camera = m_pRenderManager->createCamera("__nauDefault");
	m_Camera->setViewport(m_Viewport);

	// Init LUA
#if NAU_LUA == 1
	LuaScriptNames.clear();
	LuaFilesWithIssues.clear();
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


// ----------------------------------------------------
//		Profiling
// ----------------------------------------------------


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

#pragma region LuaStuff
// ----------------------------------------------------
//		Lua Stuff
// ----------------------------------------------------

#if NAU_LUA == 1


void printTable(lua_State *L, int pos)
{
	lua_pushnil(L);
	while (lua_next(L, -2) != 0)
	{
		if (lua_isstring(L, -1)) {
			LOG_trace_nr("%s ", lua_tostring(L, -1));
		}
		else if (lua_isnumber(L, -1)) {
			LOG_trace_nr("%f", lua_tonumber(L, -1));
		}
		else if (lua_istable(L, -1))
			printTable(L, -1);

		lua_pop(L, 1);
	}
}


void
Nau::luaStackDump(lua_State *LuaState)
{
	int i;
	int top = lua_gettop(LuaState);

	LOG_trace("LUA: total in stack %d", top);

	for (i = 1; i <= top; i++)
	{
		int t = lua_type(LuaState, i);
		switch (t) {
		case LUA_TSTRING:
			LOG_trace("LUA: string: '%s'", lua_tostring(LuaState, i));
			break;
		case LUA_TBOOLEAN:
			LOG_trace("LUA: boolean %s", lua_toboolean(LuaState, i) ? "true" : "false");
			break;
		case LUA_TNUMBER:
			LOG_trace("LUA: number: %g", lua_tonumber(LuaState, i));
			break;
		case LUA_TTABLE:
			LOG_trace_nr("LUA: table {")
			lua_pushvalue(LuaState, i);
			printTable(LuaState, i);
			lua_pop(LuaState, 1);
			LOG_trace_nr("}\n");
			break;
		default:
			LOG_trace("LUA: %s", lua_typename(LuaState, t));
			break;
		}
	}
}


void 
Nau::luaGetValues(lua_State *l, void *arr, int card, Enums::DataType bdt) {

	float *arrF;
	int *arrI;
	unsigned int *arrUI;
	double *arrD;

	switch (bdt) {

	case Enums::FLOAT:
		arrF = (float *)arr;
		for (int i = 0; i < card; ++i) {
			lua_pushnumber(l, i + 1); // key
			lua_pushnumber(l, arrF[i]); // value
			lua_settable(l, -3); // two pushes 
		}
		break;
	case Enums::DOUBLE:
		arrD = (double *)arr;
		for (int i = 0; i < card; ++i) {
			lua_pushnumber(l, i + 1); // key
			lua_pushnumber(l, arrD[i]); // value
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
Nau::luaGetBuffer(lua_State *l) {

	if (NAU->getTraceStatus()) {
		LOG_trace("LUA: Calling getBuffer");
		luaStackDump(LuaState);
	}
	int top = lua_gettop(l);
	// some validation
	if (top != 4) {
		SLOG("Lua script %s ERROR : Lua getBuffer requires 4 arguments: buffer name, offset, the attribute to set, data type, and a table to hold the return values",
			LuaCurrentScript.c_str());
		LuaFilesWithIssues.insert(LuaScriptNames[LuaCurrentScript]);
	}

	luaL_checktype(l, 1, LUA_TSTRING);
	luaL_checktype(l, 2, LUA_TNUMBER);
	luaL_checktype(l, 3, LUA_TSTRING);
	luaL_checktype(l, 4, LUA_TTABLE);

	const char *name = lua_tostring(l, -4);
	size_t offset = (size_t)lua_tointeger(l, -3);
	const char *dataType = lua_tostring(l, -2);

	Enums::DataType dt = Enums::getType(dataType);
	int card = Enums::getCardinality(dt);
	Enums::DataType bdt = Enums::getBasicType(dt);

	IBuffer *buff = RESOURCEMANAGER->getBuffer(name);
	if (buff == NULL) {
		SLOG("Lua script %s ERROR : Lua getBuffer -> invalid buffer: %s", 
			LuaCurrentScript.c_str(), name);
		LuaFilesWithIssues.insert(LuaScriptNames[LuaCurrentScript]);
		return 0;
	}

	int size = Enums::getSize(dt);
	void *arr = malloc(size);
	size_t count = buff->getData(offset, size , arr);
	if (size != count) {
		SLOG("Lua script %s ERROR : Lua getBuffer -> buffer %s offset %d, out of bounds",
			LuaCurrentScript.c_str(), name, (unsigned int)offset);
		LuaFilesWithIssues.insert(LuaScriptNames[LuaCurrentScript]);
		return 0;
	}

	luaGetValues(l, arr, card, bdt);
	free(arr);
	return 0;
}


int 
Nau::luaSetBuffer(lua_State *l) {

	if (NAU->getTraceStatus()) {
		LOG_trace("LUA: Calling setBuffer");
		luaStackDump(LuaState);
	}

	int top = lua_gettop(l);
	// some validation
	if (top != 4) {
		SLOG("Lua script %s ERROR : Lua setBuffer requires 4 arguments: buffer name, offset, the attribute to set, data type, and the values to set",
			LuaCurrentScript.c_str());
		LuaFilesWithIssues.insert(LuaScriptNames[LuaCurrentScript]);
	}

	luaL_checktype(l, 1, LUA_TSTRING);
	luaL_checktype(l, 2, LUA_TNUMBER);
	luaL_checktype(l, 3, LUA_TSTRING);
	luaL_checktype(l, 4, LUA_TTABLE);

	const char *name = lua_tostring(l, -4);
	size_t offset = (size_t)lua_tointeger(l, -3);
	const char *dataType = lua_tostring(l, -2);


	Enums::DataType dt = Enums::getType(dataType);
	int card = Enums::getCardinality(dt);
	Enums::DataType bdt = Enums::getBasicType(dt);
	int size = Enums::getSize(dt);

	IBuffer *buff = RESOURCEMANAGER->getBuffer(name);
	if (buff == NULL) {
		SLOG("Lua script %s ERROR : Lua setBuffer -> invalid buffer: %s",
			LuaCurrentScript.c_str(), name);
		LuaFilesWithIssues.insert(LuaScriptNames[LuaCurrentScript]);
		return 0;
	}

	float *arrF;
	int *arrI; 
	unsigned int *arrUI;
	double *arrD;

	switch (bdt) {

	case Enums::FLOAT:
		arrF = (float *)malloc(sizeof(float) * card);
		lua_pushnil(l);
		for (int i = 0; i < card && lua_next(l,-2) != 0; ++i) {
			arrF[i] = (float)lua_tonumber(l, -1);
			lua_pop(l, 1);
		}
		buff->setSubData(offset, size, arrF);
		free(arrF);
		break;
	case Enums::DOUBLE:
		arrD = (double *)malloc(sizeof(double) * card);
		lua_pushnil(l);
		for (int i = 0; i < card && lua_next(l, -2) != 0; ++i) {
			arrD[i] = (double)lua_tonumber(l, -1);
			lua_pop(l, 1);
		}
		buff->setSubData(offset, size, arrD);
		free(arrD);
		break;
	case Enums::INT:
	case Enums::BOOL:
		arrI = (int *)malloc(sizeof(int) * card);
		lua_pushnil(l);
		for (int i = 0; i < card && lua_next(l, -2) != 0; ++i) {
			arrI[i] = (int)lua_tointeger(l, -1);
			lua_pop(l, 1);
		}
		buff->setSubData(offset, size, arrI);
		free(arrI);
		break;
	case Enums::UINT :
		arrUI = (unsigned int *)malloc(sizeof(unsigned int) * card);
		lua_pushnil(l);
		for (int i = 0; i < card && lua_next(l, -2) != 0; ++i) {
			arrUI[i] = (unsigned int)lua_tointeger(l, -1);
			lua_pop(l, 1);
		}
		buff->setSubData(offset, size, arrUI);
		free(arrUI);
		break;
	}

	return 0;
}


int
Nau::luaGet(lua_State *l) {

	if (NAU->getTraceStatus()) {
		LOG_trace("LUA: Calling getAttr");
		luaStackDump(LuaState);
	}
	luaL_checktype(l, 1, LUA_TSTRING);
	luaL_checktype(l, 2, LUA_TSTRING);
	luaL_checktype(l, 3, LUA_TSTRING);
	luaL_checktype(l, 4, LUA_TNUMBER);
	luaL_checktype(l, 5, LUA_TTABLE);

	int top = lua_gettop(l);
	// some validation
	if (top != 5) {
		SLOG("Lua script %s ERROR : Lua getAttr requires 5 arguments: objtype, context (either the name of the object or CURRENT where applicable), the attribute to set, the index (where applicable, or 0), values that will be returned",
			LuaCurrentScript.c_str());
		LuaFilesWithIssues.insert(LuaScriptNames[LuaCurrentScript]);
	}

	const char *tipo = lua_tostring(l, -5);
	const char *context = lua_tostring(l, -4);
	const char *component = lua_tostring(l, -3);
	int number = (int)lua_tointeger(l, -2);
	void *arr;
	AttribSet *attr;


	if (!strcmp(context, "CURRENT")) {
		
		attr = NAU->getCurrentObjectAttributes(tipo)->getAttribSet();
		if (attr == NULL) {
			SLOG("Lua script %s ERROR : Lua set -> Invalid context: %s", 
				LuaCurrentScript.c_str(), context);
			LuaFilesWithIssues.insert(LuaScriptNames[LuaCurrentScript]);
		}
	}
	else {
		attr = NAU->getAttribs(tipo);
		if (attr == NULL) {
			SLOG("Lua script %s ERROR : Lua get -> invalid type: %s", 
				LuaCurrentScript.c_str(), tipo);
			LuaFilesWithIssues.insert(LuaScriptNames[LuaCurrentScript]);
		}
	}

	std::string s = component;
	Enums::DataType dt, bdt;
	int id;
	attr->getPropTypeAndId(s, &dt, &id);
	if (id == -1) {
		SLOG("Lua script %s ERROR: Lua get -> invalid component: %s", 
			LuaCurrentScript.c_str(), component);
		LuaFilesWithIssues.insert(LuaScriptNames[LuaCurrentScript]);
		lua_pushnumber(l, 1); // key
		lua_pushnumber(l, 0); // value
		lua_settable(l, -3); // two pushes 
		return 0;
	}

	int card = Enums::getCardinality(dt);
	bdt = Enums::getBasicType(dt);
	arr = NAU->getAttributeValue(tipo, context, component, number);
	if (arr == NULL) {
		SLOG("Lua script %s ERROR: Lua get -> Invalid context or number: %s %d", 
			LuaCurrentScript.c_str(), context, number);
		LuaFilesWithIssues.insert(LuaScriptNames[LuaCurrentScript]);
	}
	if (!Enums::isBasicType(dt)) {
		arr = ((Data *)(arr))->getPtr();
	}
	luaGetValues(l, arr, card, bdt);

	return 0;
}


int 
Nau::luaSet(lua_State *l) {

	if (NAU->getTraceStatus()) {
		LOG_trace("LUA: Calling setAttr");
		luaStackDump(LuaState);
	}
	int top = lua_gettop(l);
	// some validation
	if (top != 5) {
		SLOG("Lua script %s ERROR: Lua setAttr requires 5 arguments: objtype, context (either the name of the object or CURRENT where applicable), the attribute to set, the index (where applicable, or 0), and the values to set",
			LuaCurrentScript.c_str());
		LuaFilesWithIssues.insert(LuaScriptNames[LuaCurrentScript]);
	}

	luaL_checktype(l, 1, LUA_TSTRING);
	luaL_checktype(l, 2, LUA_TSTRING);
	luaL_checktype(l, 3, LUA_TSTRING);
	luaL_checktype(l, 4, LUA_TNUMBER);
	luaL_checktype(l, 5, LUA_TTABLE);

	const char *tipo = lua_tostring(l, -5);
	const char *context = lua_tostring(l, -4);
	const char *component = lua_tostring(l, -3);
	if (!lua_isnumber(l, -2)) {
		lua_pushstring(l, "The 4th argument must be the index (where applicable) or 0");
		lua_error(l);
	}
	int number = (int)lua_tointeger(l, - 2);
	Data *arr = NULL;
	AttribSet *attr;


	if (!strcmp(context, "CURRENT")) {
		try {
			AttributeValues *av = NAU->getCurrentObjectAttributes(tipo);
			attr = av->getAttribSet();
		}
		catch (std::string e) {
			SLOG("Lua script %s ERROR: Lua set -> Invalid type: %s", 
				LuaCurrentScript.c_str(), tipo);
			LuaFilesWithIssues.insert(LuaScriptNames[LuaCurrentScript]);
		}
	}
	else {
		attr = NAU->getAttribs(tipo);
		if (attr == NULL) {
			SLOG("Lua script %s ERROR: Lua set -> Invalid type: %s",
				LuaCurrentScript.c_str(), tipo);
			LuaFilesWithIssues.insert(LuaScriptNames[LuaCurrentScript]);
		}
	}
	std::string s = component;
	Enums::DataType dt, bdt;
	int id;
	attr->getPropTypeAndId(s, &dt, &id);
	if (id == -1) {
		SLOG("Lua script %s ERROR: Lua set -> Invalid component: %s", 
			LuaCurrentScript.c_str(), component);
		LuaFilesWithIssues.insert(LuaScriptNames[LuaCurrentScript]);
		return 0;
	}
	int card = Enums::getCardinality(dt);
	bdt = Enums::getBasicType(dt);
	float *arrF;
	int *arrI; 
	unsigned int *arrUI;
	double *arrD;

	switch (bdt) {

	case Enums::DOUBLE:
		arrD = (double *)malloc(sizeof(double) * card);
		lua_pushnil(l);
		for (int i = 0; i < card && lua_next(l, -2) != 0; ++i) {
			arrD[i] = (double)lua_tonumber(l, -1);
			lua_pop(l, 1);
		}
		switch (dt) {
		case Enums::DOUBLE:
			arr = new NauDouble(*arrD); break;
		case Enums::DVEC2:
			arr = new dvec2(arrD[0], arrD[1]); break;
		case Enums::DVEC3:
			arr = new dvec3(arrD[0], arrD[1], arrD[2]); break;
		case Enums::DVEC4:
			arr = new dvec4(arrD[0], arrD[1], arrD[2], arrD[3]); break;
		case Enums::DMAT3:
			arr = new dmat3(arrD); break;
		case Enums::DMAT4:
			arr = new dmat4(arrD); break;
		default:
			SLOG("Lua script %s ERROR: Lua set -> Type %s not supported",
				LuaCurrentScript.c_str(), Enums::DataTypeToString[dt].c_str());
			LuaFilesWithIssues.insert(LuaScriptNames[LuaCurrentScript]);
		}
		free(arrD);
		break;
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
			SLOG("Lua script %s ERROR: Lua set -> Type %s not supported",
				LuaCurrentScript.c_str(), Enums::DataTypeToString[dt].c_str());
			LuaFilesWithIssues.insert(LuaScriptNames[LuaCurrentScript]);
		}
		free(arrF);
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
			SLOG("Lua script %s ERROR: Lua set -> Type %s not supported",
				LuaCurrentScript.c_str(), Enums::DataTypeToString[dt].c_str());
			LuaFilesWithIssues.insert(LuaScriptNames[LuaCurrentScript]);
		}
		free(arrI);
		break;
	case Enums::UINT:
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
			SLOG("Lua script %s ERROR: Lua set -> Type %s not supported",
				LuaCurrentScript.c_str(),  Enums::DataTypeToString[dt].c_str());
			LuaFilesWithIssues.insert(LuaScriptNames[LuaCurrentScript]);
		}
		free(arrUI);
		break;
	default:
		SLOG("Lua script %s ERROR: Lua set -> Type %s not supported", 
			LuaCurrentScript.c_str(), Enums::DataTypeToString[bdt].c_str());
		LuaFilesWithIssues.insert(LuaScriptNames[LuaCurrentScript]);
	}

	try {
		NAU->setAttributeValue(tipo, context, component, number, arr);
	}
	catch (std::string e) {
		SLOG("Lua script %s ERROR: Lua set -> Invalid context: %s", 
			LuaCurrentScript.c_str(), context);
		LuaFilesWithIssues.insert(LuaScriptNames[LuaCurrentScript]);
	}

	delete arr;
	return 0;
}


int
Nau::luaSaveTexture(lua_State *l) {

	if (NAU->getTraceStatus()) {
		LOG_trace("LUA: Calling saveTexture");
		luaStackDump(LuaState);
	}

	int top = lua_gettop(l);
	if (top != 1) {
		SLOG("Lua script %s ERROR: Lua saveTexture takes a single argument: the texture name",
			LuaCurrentScript.c_str());
		LuaFilesWithIssues.insert(LuaScriptNames[LuaCurrentScript]);
	}

	const char *texName = lua_tostring(l, -1);

	if (!RESOURCEMANAGER->hasTexture(texName)) {
		SLOG("Lua script %s ERROR: Lua save texture -> invalid texture name: %s",
			LuaCurrentScript.c_str(), texName);
		LuaFilesWithIssues.insert(LuaScriptNames[LuaCurrentScript]);
	}

	nau::material::ITexture *texture = RESOURCEMANAGER->getTexture(std::string(texName));

	char s[200];
	sprintf(s, "%s.%d.png", texture->getLabel().c_str(), RENDERER->getPropui(IRenderer::FRAME_COUNT));
	std::string sname = nau::system::File::Validate(s);
	ITextureLoader::Save(texture, ITextureLoader::PNG);

	return 0;
}


int
Nau::luaSaveBuffer(lua_State *l) {

	if (NAU->getTraceStatus()) {
		LOG_trace("LUA: Calling saveBuffer");
		luaStackDump(LuaState);
	}

	int top = lua_gettop(l);
	if (top != 1) {
		SLOG("Lua script %s ERROR: Lua saveBuffer takes a single argument: the buffer name",
			LuaCurrentScript.c_str());
		LuaFilesWithIssues.insert(LuaScriptNames[LuaCurrentScript]);
	}

	const char *bufferName = lua_tostring(l, -1);

	if (!RESOURCEMANAGER->hasBuffer(bufferName)) {
		SLOG("Lua script %s ERROR: Lua save buffer -> invalid buffer name: %s", 
			LuaCurrentScript.c_str(), bufferName);
		LuaFilesWithIssues.insert(LuaScriptNames[LuaCurrentScript]);
	}

	nau::material::IBuffer *buffer = RESOURCEMANAGER->getBuffer(std::string(bufferName));

	BufferLoader::SaveBuffer(buffer);

	return 0;
}


int
Nau::luaSaveProfile(lua_State *l) {

	if (NAU->getTraceStatus()) {
		LOG_trace("LUA: call saveProfile");
		luaStackDump(LuaState);
	}
	const char *filename = lua_tostring(l, -1);

	std::string prof;
	Profile::DumpLevels(prof);

	fstream s;
	s.open(filename, fstream::out);
	s << prof << "\n";
	s.close();

	NAU->setProfileResetRequest();
	//Profile::Reset();
	return 0;
}


int
Nau::luaScreenshot(lua_State* l) {

	RENDERER->saveScreenShot();
	return 0;
}


bool 
Nau::luaCheckScriptName(std::string fileName, std::string scriptName) {

	if (LuaScriptNames.count(scriptName) && LuaScriptNames[scriptName] != fileName) {
		return false;
	}

	LuaScriptNames[scriptName] = fileName;
	return true;
}


void 
luaDebug(lua_State *LuaState) {

	lua_Debug info;
	int level = 0;
	int x = lua_getstack(LuaState, level, &info);
	if (lua_getstack(LuaState, level, &info) != 1) {
		lua_getinfo(LuaState, "nSl", &info);
		fprintf(stderr, "  [%d] %s:%d -- %s [%s]\n",
			level, info.short_src, info.currentline,
			(info.name ? info.name : "<unknown>"), info.what);
		++level;
	}
}


void
Nau::initLua() {

	LuaState = luaL_newstate();
	luaL_openlibs(LuaState);
	lua_pushcfunction(LuaState, Nau::luaSet);
	lua_setglobal(LuaState, "setAttr");
	lua_pushcfunction(LuaState, Nau::luaGet);
	lua_setglobal(LuaState, "getAttr");
	lua_pushcfunction(LuaState, Nau::luaGetBuffer);
	lua_setglobal(LuaState, "getBuffer");
	lua_pushcfunction(LuaState, Nau::luaSaveTexture);
	lua_setglobal(LuaState, "saveTexture");
	lua_pushcfunction(LuaState, Nau::luaSetBuffer);
	lua_setglobal(LuaState, "setBuffer");
	lua_pushcfunction(LuaState, Nau::luaSaveProfile);
	lua_setglobal(LuaState, "saveProfiler");
	lua_pushcfunction(LuaState, Nau::luaScreenshot);
	lua_setglobal(LuaState, "screenshot");
}


void
Nau::initLuaScript(std::string file, std::string name) {

	//std::string fileName = File::GetName(file);

	if (LuaFilesWithIssues.find(name) != LuaFilesWithIssues.end())
		return;

	int res = luaL_dofile(LuaState, file.c_str());
	if (res) {
		SLOG("Error loading lua file: %s", lua_tostring(LuaState, -1));
		LuaFilesWithIssues.insert(name);
	}
}


void 
Nau::compileLuaScripts() {

	LuaFilesWithIssues.clear();

	std::set<std::string> luaFiles;
	for (auto &s : LuaScriptNames) {
			luaFiles.insert(s.second);
	}
	for (auto &s : luaFiles) {
		initLuaScript(s, "");
	}

	if (LuaFilesWithIssues.size() == 0) {
		SLOG("Lua Compilation OK");
	}

}


void
Nau::callLuaScript(std::string name) {

	if (LuaFilesWithIssues.size() != 0 &&
		LuaFilesWithIssues.find(LuaScriptNames[name]) != LuaFilesWithIssues.end())
		return;

	{
		PROFILE("Lua");
		LuaCurrentScript = name;

		int errIndex = 0;

		if (m_TraceOn) {
			LOG_trace("#LUA %s", name.c_str());

			lua_getglobal(LuaState, "debug");
			lua_getfield(LuaState, -1, "traceback");
			lua_remove(LuaState, -2);
			errIndex = -2;
		}
		lua_getglobal(LuaState, name.c_str());

		// do we have an error?
		if (lua_pcall(LuaState, 0, 0, errIndex)) {
			//	if (lua_pcall(LuaState, 0, 0, 0)) {
					//luaL_traceback(LuaState, LuaState, "hello", 0);
					//lua_Debug info;
					//int level = 0;

					//while (lua_getstack(LuaState, level, &info)) {
					//	lua_getinfo(LuaState, "nSl", &info);
					//	SLOG("  [%d] %s:%d -- %s [%s]\n",
					//		level, info.short_src, info.currentline,
					//		(info.name ? info.name : "<unknown>"), info.what);
					//	++level;
					//}
			if (!lua_isnil(LuaState, -1)) {
				const char *msg = lua_tostring(LuaState, -1);
				if (msg != NULL) {
					SLOG("Lua script %s ERROR: %s", name.c_str(), msg);
					LuaFilesWithIssues.insert(LuaScriptNames[name]);
				}
			}

		}
		if (m_TraceOn) {
			lua_pop(LuaState, 1);
		}

		LuaCurrentScript = "";
	}
}


bool
Nau::callLuaTestScript(std::string name) {

	// if script has errors then just return false
	if (LuaFilesWithIssues.size() != 0 &&
		LuaFilesWithIssues.find(LuaScriptNames[name]) != LuaFilesWithIssues.end())
		return false;

	{
		PROFILE("LUA");
		LuaCurrentScript = name;

		int errIndex = 0;

		if (m_TraceOn) {
			LOG_trace("#LUA: call script %s", name.c_str());

			lua_getglobal(LuaState, "debug");
			lua_getfield(LuaState, -1, "traceback");
			lua_remove(LuaState, -2);
			errIndex = -2;
		}

		lua_getglobal(LuaState, name.c_str());

		if (lua_pcall(LuaState, 0, 1, errIndex)) {
			if (!lua_isnil(LuaState, -1)) {
				const char *msg = lua_tostring(LuaState, -1);
				if (msg != NULL) {
					SLOG("Lua script %s ERROR: %s", name.c_str(), msg);
					LuaFilesWithIssues.insert(LuaScriptNames[name]);
				}
			}
		}

		int result = lua_toboolean(LuaState, -1);
		lua_pop(LuaState, 1);

		if (m_TraceOn) {
			lua_pop(LuaState, 1);
		}
		return (result != 0);
		LuaCurrentScript = "";
	}
}


#endif


#pragma endregion





// ----------------------------------------------------------
//		GET AND SET ATTRIBUTES
// ----------------------------------------------------------


std::unique_ptr<Attribute>& 
Nau::getAttribute(const std::string & type, const std::string & component) {

	// type is assumed to be valid. Use validateObjectType to check
	assert(m_Attributes.count(type));
	return m_Attributes[type]->get(component);
}


AttributeValues *
Nau::getCurrentObjectAttributes(const std::string &type, int number) {

	IRenderer *renderer = m_pRenderManager->getRenderer();
	IAPISupport *sup = IAPISupport::GetInstance();

	if (type == "CAMERA") {
		return (AttributeValues *)renderer->getCamera().get();
	}
	else if (type == "COLOR") {
		return (AttributeValues *)renderer->getColorMaterial();
	}
	
	else if (sup->apiSupport(IAPISupport::APIFeatureSupport::IMAGE_TEXTURE) && type == "IMAGE_TEXTURE") {
		return (AttributeValues *)renderer->getImageTexture(number);
	}

	else if (type == "LIGHT") {
		return (AttributeValues *)renderer->getLight(number).get();
	}
	else if (type == "BUFFER_BINDING") {
		return (AttributeValues *)renderer->getMaterial()->getMaterialBuffer(number);
	}
	else if (type == "BUFFER_MATERIAL") {
		return (AttributeValues *)renderer->getMaterial()->getBuffer(number);
	}
	else if (type == "IMAGE_TEXTURE") {
		return (AttributeValues *)renderer->getMaterial()->getImageTexture(number);
	}
	else if (type == "ARRAY_OF_IMAGE_TEXTURES") {
		return (AttributeValues *)renderer->getMaterial()->getArrayOfImageTextures(number);
	}
	else if (type == "TEXTURE_BINDING") {
		return (AttributeValues *)renderer->getMaterialTexture(number);
	}
	else if (type == "PASS") {
		return (AttributeValues *)m_pRenderManager->getCurrentPass();
	}
	else if (type == "RENDERER") {
		return (AttributeValues *)renderer;
	}
	else if (type == "RENDER_TARGET") {
		return (AttributeValues *)m_pRenderManager->getCurrentPass()->getRenderTarget();
			//m_pResourceManager->getRenderTarget(context);
	}
	else if (type == "STATE") {
		return (AttributeValues *)renderer->getState();
	}
	else if (type == "TEXTURE") {
		return (AttributeValues *)renderer->getTexture(number);
	}
	else if (type == "VIEWPORT") {
		return (AttributeValues *)renderer->getViewport().get();
	}
	else if (type == "ARRAY_OF_TEXTURES_BINDING") {
		return (AttributeValues *)renderer->getMaterial()->getMaterialArrayOfTextures(number);
	}
	// If we get here then we are trying to fetch something that does not exist
	NAU_THROW("Getting an invalid object\ntype: %s", type.c_str());
}



AttributeValues *
Nau::getObjectAttributes(const std::string &type, const std::string &context, int number) {

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

	if (type == "PASS") {
		std::string pipName = m_pRenderManager->getActivePipelineName();
		if (m_pRenderManager->hasPass(pipName, context))
			return (AttributeValues *)m_pRenderManager->getPass(pipName, context);
	}

	if (type == "RENDERER") {
		return (AttributeValues *)RENDERER;
	}

	if (type == "SCENE") {
		if (m_pRenderManager->hasScene(context))
			return (AttributeValues *)m_pRenderManager->getScene(context).get();
	}

	if (type == "SPHERE" || type == "GRID") {
		std::string scene, object;
		std::size_t found = context.find("::");
		if (found != std::string::npos && context.size() > found + 2) {
			scene = context.substr(0, found);
			object = context.substr(found + 2);
		}

		if (m_pRenderManager->hasScene(scene)) {
			SceneObject *s = m_pRenderManager->getScene(scene)->getSceneObject(object).get();
			if (s) {
				IRenderable *r = s->getRenderable().get();
				std::string s = r->getClassName();
				std::transform(s.begin(), s.end(), s.begin(), ::toupper);
				//std::string s = typeid(*r).name();
				if (type == s) {
					//Sphere *sp = (Sphere *)r;
					return (AttributeValues *)r;

				}
			}
		}
	}

	if (type == "PASS_POST_PROCESS_TEXTURE") {
		Pass *p = RENDERMANAGER->getPass(context);
		if (p) {
			PassProcessItem *pp = p->getPostProcessItem(number);
			std::string s = typeid(*pp).name();
			if (s == "class nau::render::PassProcessTexture") {
				PassProcessTexture *ppt = (PassProcessTexture *)pp;
				return (AttributeValues *)ppt;
			}
		}
	}
	if (type == "PASS_PRE_PROCESS_TEXTURE") {
		Pass *p = RENDERMANAGER->getPass(context);
		if (p) {
			PassProcessItem *pp = p->getPreProcessItem(number);
			std::string s = typeid(*pp).name();
			if (s == "class nau::render::PassProcessTexture") {
				PassProcessTexture *ppt = (PassProcessTexture *)pp;
				return (AttributeValues *)ppt;
			}
		}
	}
	if (type == "PASS_POST_PROCESS_BUFFER") {
		Pass *p = RENDERMANAGER->getPass(context);
		if (p) {
			PassProcessItem *pp = p->getPostProcessItem(number);
			std::string s = typeid(*pp).name();
			if (s == "class nau::render::PassProcessBuffer") {
				PassProcessBuffer *ppt = (PassProcessBuffer *)pp;
				return (AttributeValues *)ppt;
			}
		}
	}
	if (type == "PASS_PRE_PROCESS_BUFFER") {
		Pass *p = RENDERMANAGER->getPass(context);
		if (p) {
			PassProcessItem *pp = p->getPreProcessItem(number);
			std::string s = typeid(*pp).name();
			if (s == "class nau::render::PassProcessBuffer") {
				PassProcessBuffer *ppt = (PassProcessBuffer *)pp;
				return (AttributeValues *)ppt;
			}
		}
	}

	if (type == "SCENE_OBJECT") {
		std::string scene, object;
		std::size_t found = context.find("::");
		if (found != std::string::npos && context.size() > found + 2) {
			scene = context.substr(0, found);
			object = context.substr(found + 2);
		}

		if (m_pRenderManager->hasScene(scene)) {
			SceneObject *s = m_pRenderManager->getScene(scene)->getSceneObject(object).get();
			if (s)
				return (AttributeValues *)s;
		}
	}

	if (type == "VIEWPORT") {
		if (m_pRenderManager->hasViewport(context))
			return (AttributeValues *)m_pRenderManager->getViewport(context).get();
	}

	// From ResourceManager

	if (type == "BUFFER") {
		if (m_pResourceManager->hasBuffer(context))
			return (AttributeValues *)m_pResourceManager->getBuffer(context);
	}
	if (type == "TEXTURE") {
		if (m_pResourceManager->hasTexture(context))
			return (AttributeValues *)m_pResourceManager->getTexture(context);
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

	if (type == "COLOR") {
		if (m_pMaterialLibManager->hasMaterial(lib, mat))
			return (AttributeValues *)&(m_pMaterialLibManager->getMaterial(lib, mat)->getColor());
	}
	if (type == "TEXTURE_MATERIAL") {
		if (m_pMaterialLibManager->hasMaterial(lib, mat))
			return (AttributeValues *)m_pMaterialLibManager->getMaterial(lib, mat)->getTexture(number);
	}
	if (type == "TEXTURE_SAMPLER") {
		if (m_pMaterialLibManager->hasMaterial(lib, mat))
			return (AttributeValues *)m_pMaterialLibManager->getMaterial(lib, mat)->getTextureSampler(number);
	}
	if (type == "TEXTURE_BINDING") {
		if (m_pMaterialLibManager->hasMaterial(lib, mat))
			return (AttributeValues *)m_pMaterialLibManager->getMaterial(lib, mat)->getMaterialTexture(number);
	}
	if (sup->apiSupport(IAPISupport::APIFeatureSupport::IMAGE_TEXTURE) && type == "IMAGE_TEXTURE") {
		if (m_pMaterialLibManager->hasMaterial(lib, mat))
			return (AttributeValues *)m_pMaterialLibManager->getMaterial(lib, mat)->getImageTexture(number);
	}
	if (type == "ARRAY_OF_IMAGE_TEXTURES") {
		if (m_pMaterialLibManager->hasMaterial(lib, mat))
			return (AttributeValues *)m_pMaterialLibManager->getMaterial(lib, mat)->getArrayOfImageTextures(number);
	}
	if (type == "BUFFER_MATERIAL") {
		if (m_pMaterialLibManager->hasMaterial(lib, mat)) {
			return (AttributeValues *)m_pMaterialLibManager->getMaterial(lib, mat)->getBuffer(number);
		}
	}
	if (type == "BUFFER_BINDING") {
		if (m_pMaterialLibManager->hasMaterial(lib, mat))
			return (AttributeValues *)m_pMaterialLibManager->getMaterial(lib, mat)->getMaterialBuffer(number);
	}
	else if (type == "ARRAY_OF_TEXTURES_BINDING") {
		if (m_pMaterialLibManager->hasMaterial(lib, mat))
			return (AttributeValues *)m_pMaterialLibManager->getMaterial(lib, mat)->getMaterialArrayOfTextures(number);
	}
	// If we get here then we are trying to fetch something that does not exist
	NAU_THROW("Getting an invalid object\ntype: %s\ncontext: %s", 
			type.c_str(), context.c_str());
}


bool
Nau::validateShaderAttribute(std::string type, std::string context, std::string component) {

	int id;
	Enums::DataType dt;

	if (m_Attributes.count(type) == 0)
		return false;

	m_Attributes[type]->getPropTypeAndId(component, &dt, &id);
	return (id != -1);
}


Enums::DataType 
Nau::getAttributeDataType(std::string type, std::string context, std::string component) {

	int id;
	Enums::DataType dt;

	if (m_Attributes.count(type) == 0)
		return Enums::COUNT_DATATYPE;

	m_Attributes[type]->getPropTypeAndId(component, &dt, &id);
	if (id != -1)
		return dt;
	else
		return Enums::COUNT_DATATYPE;
}


bool 
Nau::setAttributeValue(std::string type, std::string context, std::string component, int number, Data *values) {

	int id;
	Enums::DataType dt; 
	AttributeValues *attrVal;

	if (context != "CURRENT") {
		m_Attributes[type]->getPropTypeAndId(component, &dt, &id);
		attrVal = NAU->getObjectAttributes(type, context, number);
	}
	else {
		m_Attributes[type]->getPropTypeAndId(component, &dt, &id);
		attrVal = NAU->getCurrentObjectAttributes(type, number);
	}

	if (attrVal == NULL || id == -1) {
		return false;
	}
	else {
		attrVal->setProp(id, dt, values);
		return true;
	}
}


void *
Nau::getAttributeValue(std::string type, std::string context, std::string component, int number) {

	int id;
	Enums::DataType dt;
	AttributeValues *attrVal;

	if (context != "CURRENT") {
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


bool 
Nau::validateObjectType(const std::string &type) {

	if (m_Attributes.count(type) == 0)
		return false;
	else
		return true;
}


void
Nau::getValidObjectTypes(std::vector<std::string> *v) {

	for (auto &s : m_Attributes)
		v->push_back(s.first);
}


void
Nau::getValidObjectNames(const std::string &type, std::vector<std::string> *v) {

	if (type == "VIEWPORT" || type == "viewport") {
		RENDERMANAGER->getViewportNames(v);
	}
	if (type == "CAMERA" || type == "camera") {
		RENDERMANAGER->getCameraNames(v);
	}
	if (type == "BUFFER" || type == "buffer") {
		return RESOURCEMANAGER->getBufferNames(v);
	}
}


bool
Nau::validateObjectName(const std::string &type, const std::string &v) {

	if (type == "")
		return true;
	if (type == "VIEWPORT" || type == "viewport") {
		return RENDERMANAGER->hasViewport(v);
	}
	if (type == "CAMERA" || type == "camera") {
		return RENDERMANAGER->hasCamera(v);
	}
	if (type == "BUFFER" || type == "buffer") {
		return RESOURCEMANAGER->hasBuffer(v);
	}
	return false;
}


AttributeValues *
Nau::createObject(const std::string &type, const std::string &name) {

	if (type == "VIEWPORT" || type == "viewport") {
		return RENDERMANAGER->createViewport(name).get();
	}
	if (type == "LIGHT" || type == "light") {
		return RENDERMANAGER->createLight(name, "default").get();
	}
	if (type == "CAMERA" || type == "camera") {
		return RENDERMANAGER->createCamera(name).get();
	}
	if (type == "BUFFER" || type == "buffer") {
		return RESOURCEMANAGER->createBuffer(name);
	}
	return NULL;
}


bool 
Nau::validateObjectContext(const std::string &type, const std::string &context) {

	if (m_Attributes.count(type) == 0)
		return false;

	AttributeValues *attrVal;

	if (context != "CURRENT") {
		attrVal = NAU->getObjectAttributes(type, context, 0);
	}
	else {
		attrVal = NAU->getCurrentObjectAttributes(type, 0);
	}

	if (attrVal == NULL)
		return false;
	else
		return true;
}


bool
Nau::validateObjectComponent(const std::string &type, const std::string & component) {

	if (m_Attributes.count(type) == 0)
		return false;

	AttribSet *as = m_Attributes[type];
	if (as->getID(component) == -1)
		return false;

	return true;
}


void
Nau::getValidObjectComponents(const std::string &type, std::vector<std::string> *v) {

	if (m_Attributes.count(type) == 0)
		return;

	AttribSet *as = m_Attributes[type];
	for (auto &attr : as->getAttributes())
		v->push_back(attr.first);
}


// -----------------------------------------------------------
//		USER ATTRIBUTES
// -----------------------------------------------------------


void 
Nau::registerAttributes(std::string s, AttribSet *attrib) {

	m_Attributes[s] = attrib;
}


bool 
Nau::validateUserAttribType(std::string type) {

	if (m_Attributes.count(type) != 0)
		return true;
	else
		return false;
}


AttribSet *
Nau::getAttribs(std::string context) {

	if (m_Attributes.count(context) != 0)
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


void 
Nau::getObjTypeList(std::vector<std::string> *v) {

	for (auto attr : m_Attributes) {
		v->push_back(attr.first);
	}
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


// -----------------------------------------------------------
//		KEYBOARD AND MOUSE
// -----------------------------------------------------------


int 
Nau::keyPressed(int key, int modifiers) {

	return 0;
}


int 
Nau::mouseButton(Nau::MouseAction action, Nau::MouseButton buttonID, int x, int y) {

	ivec2 iv2(x, y);
	if (action == PRESSED) {
		switch (buttonID) {
		case LEFT:
			RENDERER->setPropi2(IRenderer::MOUSE_LEFT_CLICK, iv2);
			break;
		case MIDDLE:
			RENDERER->setPropi2(IRenderer::MOUSE_MIDDLE_CLICK, iv2);
			break;
		case RIGHT:
			RENDERER->setPropi2(IRenderer::MOUSE_RIGHT_CLICK, iv2);
			break;
		}
	}

	return 0;

}


int 
Nau::mouseMotion(int x, int y) {

	return 0;
}


int 
Nau::mouseWheel(int pos) {

	return 0;
}


// -----------------------------------------------------------
//		LOAD ASSETS
// -----------------------------------------------------------


void
Nau::clear() {

	RENDERER->setPropui(IRenderer::FRAME_COUNT, 0);
	RENDERER->m_AtomicLabels.clear();
	setActiveCameraName("__nauDefault");
	SceneObject::ResetCounter();
	MATERIALLIBMANAGER->clear();
	EVENTMANAGER->clear();
	EVENTMANAGER->addListener("WINDOW_SIZE_CHANGED", this);
	RENDERMANAGER->clear();
	RESOURCEMANAGER->clear();
	Profile::Reset();
	deleteUserAttributes();
	UniformBlockManager::DeleteInstance();
	m_pPhysicsManager->clear();
	m_Physics = false;

	m_Viewport = RENDERMANAGER->createViewport("__nauDefault");

	ProjectLoader::loadMatLib(m_AppFolder + File::PATH_SEPARATOR + "nauSettings/nauSystem.mlib");

	INTERFACE_MANAGER->clear();

#if NAU_LUA == 1
	LuaScriptNames.clear();
	LuaFilesWithIssues.clear();
#endif
}


void
Nau::readProjectFile(std::string file, int *width, int *height) {

	m_ProjectFolder = File::GetPath(file);
	try {
		ProjectLoader::load(file, width, height);
	}
	catch (std::string s) {
		clear();
		throw(s);
	}

	setActiveCameraName(RENDERMANAGER->getDefaultCameraName());
}


void
Nau::readDirectory (std::string dirName) {

	clear();
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
	loadFilesAndFoldersAux(dirName, false);	
}

void 
Nau::saveProject(std::string filename) {

	ProjectLoader::saveProject(filename);
}


void
Nau::readModel (std::string filename) throw (std::string) {

	clear();
	bool result = true;
	
	char sceneName[256];

	sprintf(sceneName,"MainScene"); //,loadedScenes);
	loadedScenes++;

	RENDERMANAGER->createScene (sceneName);

	try {
		NAU->loadAsset (filename, sceneName);
		loadFilesAndFoldersAux(sceneName, false);
	} 
	catch (std::string &s) {
			throw(s);
	}
}


void
Nau::appendModel(std::string filename) {
	
	char sceneName[256];

	sprintf(sceneName,"MainScene"); //,loadedScenes);
	loadedScenes++;

	RENDERMANAGER->createScene (sceneName);

	try {
		NAU->loadAsset (filename, sceneName);
		loadFilesAndFoldersAux(sceneName, false);
	}
	catch( std::string &s){
		throw(s);
	}
}


void Nau::loadFilesAndFoldersAux(std::string sceneName, bool unitize) {

	m_ProjectName = "Model Display";
	Camera *aNewCam = m_pRenderManager->getCamera ("MainCamera").get();
	std::shared_ptr<Viewport> v = m_pRenderManager->getViewport("__nauDefault");//createViewport ("MainViewport", nau::math::vec4(0.0f, 0.0f, 0.0f, 1.0f));
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

	std::shared_ptr<Pipeline> &aPipeline = m_pRenderManager->createPipeline("MainPipeline");
	Pass *aPass = aPipeline->createPass("MainPass");
	aPass->setCamera ("MainCamera");

	aPass->addViewport (v);
	aPass->setPropb(Pass::COLOR_CLEAR, true);
	aPass->setPropb(Pass::DEPTH_CLEAR, true);
	aPass->addLight ("MainDirectionalLight");

	aPass->addScene(sceneName);
	m_pRenderManager->setActivePipeline("MainPipeline");
//	RENDERMANAGER->prepareTriangleIDsAndTangents(true,true);
}


std::string 
Nau::getProjectFolder() {

	return m_ProjectFolder;
}

// -----------------------------------------------------------
//		RUN & DEBUG
// -----------------------------------------------------------


void 
Nau::setTrace(int frames) {

	m_TraceFrames = frames;
	m_TraceOn = m_TraceFrames != 0;
}


bool
Nau::getTraceStatus() {

	return m_TraceOn;
}


void 
Nau::step() {
	if (m_ProjectName == "" || RENDERMANAGER->getNumPipelines() == 0)
		return;

	IRenderer *renderer = RENDERER;
	if (!renderer)
		return;

	float timer = (float)(clock() * INV_CLOCKS_PER_MILISEC);
	if (NO_TIME == m_LastFrameTime) {
		m_LastFrameTime = timer;
	}
	double deltaT = timer - m_LastFrameTime;
	m_LastFrameTime = timer;

	m_TraceOn = m_TraceFrames != 0;
	m_TraceFrames = renderer->setTrace(m_TraceFrames);
	if (m_TraceOn) {
		LOG_trace("#NAU(FRAME,START)");
	}

	m_pEventManager->notifyEvent("FRAME_BEGIN", "Nau", "", NULL);


	renderer->resetCounters();
	RESOURCEMANAGER->clearBuffers();

	unsigned char pip_changed;

	pip_changed = m_pRenderManager->renderActivePipeline();

	m_pEventManager->notifyEvent("FRAME_END", "Nau", "", NULL);

	if (!pip_changed) {
		unsigned int k = renderer->getPropui(IRenderer::FRAME_COUNT);
		if (k == UINT_MAX)
			// 2 avoid issues with run_once and skip_first
			// and allows a future implementation of odd and even frames for
			// ping-pong rendering
			renderer->setPropui(IRenderer::FRAME_COUNT, 2);
		else
			renderer->setPropui(IRenderer::FRAME_COUNT, ++k);
	}

	if (m_Physics)
		m_pPhysicsManager->update();

	INTERFACE_MANAGER->render();	
}


void Nau::stepPass() {

	IRenderer *renderer = RENDERER;
	float timer =(float)(clock() * INV_CLOCKS_PER_MILISEC);
	if (NO_TIME == m_LastFrameTime) {
		m_LastFrameTime = timer;
	}
	double deltaT = timer - m_LastFrameTime;
	m_LastFrameTime = timer;

	std::shared_ptr<Pipeline> &p = RENDERMANAGER->getActivePipeline();
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

		renderer->resetCounters();

		if (true == m_Physics) {
			m_pPhysicsManager->update();
//			m_pWorld->update();
//			m_pEventManager->notifyEvent("DYNAMIC_CAMERA", "MainCanvas", "", NULL);
		}

	}

 	std::string s = RENDERMANAGER->getCurrentPass()->getName();
	//if (m_TraceFrames) {
	//	LOG_trace("#NAU(PASS START %s)", s.c_str());
	//}

	p->executeNextPass();

	//if (m_TraceFrames) {
	//	LOG_trace("#NAU(PASS END %s)", s.c_str());
	//}

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

	std::shared_ptr<Pipeline> &p = RENDERMANAGER->getActivePipeline();
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
	else // 
		return m_Camera.get();
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
Nau::loadAsset (std::string aFilename, std::string sceneName, std::string params) throw (std::string) {

	File file (aFilename);
	std::string fullPath = file.getFullPath();
	try {
		switch (file.getType()) {
			case File::PATCH:
				PatchLoader::loadScene(RENDERMANAGER->getScene (sceneName).get(), fullPath);
				break;
			case File::COLLADA:
			case File::FBX:
			case File::BLENDER:
			case File::PLY:
			case File::LIGHTWAVE:
			case File::STL:
			case File::TRUESPACE:
				AssimpLoader::loadScene(RENDERMANAGER->getScene (sceneName).get(), fullPath,params);
				break;
			case File::NAUBINARYOBJECT:
				CBOLoader::loadScene(RENDERMANAGER->getScene(sceneName).get(), fullPath, params);
				break;
			case File::THREEDS:
				AssimpLoader::loadScene(RENDERMANAGER->getScene (sceneName).get(), fullPath, params);
				//THREEDSLoader::loadScene (RENDERMANAGER->getScene (sceneName), file.getFullPath(),params);				
				break;
			case File::WAVEFRONTOBJ:
				//AssimpLoader::loadScene(RENDERMANAGER->getScene (sceneName).get(), fullPath,params);
				OBJLoader::loadScene(RENDERMANAGER->getScene (sceneName).get(), fullPath, params);
				break;
			case File::OGREXMLMESH:
				OgreMeshLoader::loadScene(RENDERMANAGER->getScene (sceneName).get(), fullPath);
				break;
			default:
				SLOG("Model loading: Unsupported file type %s\n", fullPath.c_str());
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
		CBOLoader::writeScene (RENDERMANAGER->getScene (sceneName).get(), aFilename);
	}
	else if (0 == fileType.compare("OBJ")) {
		OBJLoader::writeScene(RENDERMANAGER->getScene(sceneName).get(), aFilename);
	}
}


void
Nau::setWindowSize (unsigned int width, unsigned int height) {

	m_WindowWidth = width;
	m_WindowHeight = height;
	vec2 v2((float)m_WindowWidth, (float)m_WindowHeight);
	m_Viewport->setPropf2(Viewport::SIZE, v2);

}


unsigned int 
Nau::getWindowHeight() {

	return(m_WindowHeight);
}


unsigned int 
Nau::getWindowWidth() {

	return(m_WindowWidth);
}


std::shared_ptr<Viewport>
Nau::getDefaultViewport() {
	
	return m_Viewport;
}


std::shared_ptr<Camera>
Nau::getDefaultCamera() {

	return m_Camera;
}


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
Nau::getRenderer(void) {

	RenderManager *r = getRenderManager();
	if (r)
		return r->getRenderer();
	else
		return NULL;
}


nau::physics::PhysicsManager *
Nau::getPhysicsManager() {

	return m_pPhysicsManager;
}


IAPISupport *
Nau::getAPISupport(void) {

	return m_pAPISupport;
}
