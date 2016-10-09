#include "physXLuaManager.h"



PhysXLuaManager::PhysXLuaManager() {
	isLoaded = false;
	//initLua(filename);
	luaState = luaL_newstate();
	luaL_openlibs(luaState);
	//lua_pushcfunction(luaState, luaSet);
	//lua_setglobal(luaState, "setAttr");
}


PhysXLuaManager::~PhysXLuaManager()
{
}

void PhysXLuaManager::initLua(std::string filename) {
	//isLoaded = !(luaL_dofile(luaState, filename.c_str()));
	isLoaded = false;
}

void PhysXLuaManager::createNewParticles(std::string funcName) {
	if (isLoaded) {
		lua_getglobal(luaState, funcName.c_str());
		lua_pcall(luaState, 0, LUA_MULTRET, 0);
		currentNewParticles.nbParticle = (int)(lua_tonumber(luaState, -1));
		lua_pop(luaState, 1);
		currentNewParticles.positions = (float*)malloc(sizeof(float)*(currentNewParticles.nbParticle*3));
		lua_pushnil(luaState);
		for (int i = 0; i < (currentNewParticles.nbParticle*3) && lua_next(luaState, -2) != 0; ++i) {
			currentNewParticles.positions[i] = (float)lua_tonumber(luaState, -1);
			lua_pop(luaState, 1);
		}
		int n = lua_gettop(luaState);
		lua_pop(luaState, n);
	}
	else {
		currentNewParticles.nbParticle = 0;
		currentNewParticles.positions = NULL;
	}
}
