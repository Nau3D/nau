// Marta
#ifndef LIGHTFACTORY_H
#define LIGHTFACTORY_H

#include "nau/scene/light.h"

namespace nau
{
	namespace scene
	{
		class LightFactory
		{
		public:
			static Light *create ( std::string lName, std::string lType);
		private:
			LightFactory(void) {};
		};
	};
};

#endif //LIGHTFACTORY_H