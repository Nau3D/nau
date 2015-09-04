#include "nau/world/worldFactory.h"

#include "nau/world/bulletWorld.h"

using namespace nau::world;

nau::world::IWorld* 
WorldFactory::create (std::string type)
{
	if ("Bullet" == type) {
		return new BulletWorld;
	}

	return 0;
}
