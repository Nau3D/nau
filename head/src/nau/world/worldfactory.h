#ifndef WORLDFACTORY_H
#define WORLDFACTORY_H

#include <nau/world/iworld.h>

namespace nau
{
	namespace world
	{
		class WorldFactory
		{
		public:
			static nau::world::IWorld* create (std::string type);
		private:
			WorldFactory(void) {};
			~WorldFactory(void) {};
		};
	};
};
#endif
