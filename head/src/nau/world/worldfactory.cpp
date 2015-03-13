#include "nau/world/worldfactory.h"

#include "nau/world/bulletworld.h"

using namespace nau::world;

nau::world::IWorld* 
WorldFactory::create (std::string type)
{
	if ("Bullet" == type) {
		return new BulletWorld;
	}

	return 0;
}
