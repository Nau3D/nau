// Marta
#ifndef LIGHTFACTORY_H
#define LIGHTFACTORY_H

#include "nau/scene/light.h"

#include "nau/render/renderManager.h"

namespace nau
{
	namespace scene
	{
		class LightFactory
		{
			friend class nau::render::RenderManager;

		public:
		private:
			static Light *create ( std::string lName, std::string lType);
			LightFactory(void) {};
		};
	};
};

#endif //LIGHTFACTORY_H