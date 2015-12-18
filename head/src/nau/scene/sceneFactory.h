#ifndef SCENEFACTORY_H
#define SCENEFACTORY_H

#include "nau/scene/iScene.h"

namespace nau
{
	namespace scene
	{
		class SceneFactory
		{
		public:
			static IScene * Create (std::string scene);
		private:
			SceneFactory(void) {};
		};
	};
};

#endif //SCENEFACTORY_H
