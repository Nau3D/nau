#ifndef _PHYSXLUAMANAGER_H
#define _PHYSXLUAMANAGER_H

#include <string>


extern "C" {
#include <lua/lua.h>
#include <lua/lauxlib.h>
#include <lua/lualib.h>
#include <lua\lua.hpp>
#include <lua\luaconf.h>
}

class PhysXLuaManager
{
public:

	typedef struct NewParticles {
		int nbParticle;
		float * positions;
	} newParticles;

	PhysXLuaManager();
	~PhysXLuaManager();

	void initLua(std::string filename);

	NewParticles getCurrentNewParticles() { return currentNewParticles; }
	void createNewParticles(std::string funcName);

	bool isFileLoaded() { return isLoaded; };

private:
	lua_State * luaState;

protected:
	bool isLoaded;
	NewParticles currentNewParticles;

};

#endif