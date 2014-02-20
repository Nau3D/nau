#ifndef SCENEFACTORY_H
#define SCENEFACTORY_H

#include <nau/scene/iscene.h>

namespace nau
{
	namespace scene
	{
		class SceneFactory
		{
		public:
			static IScene * create (std::string scene);
		private:
			SceneFactory(void) {};
		};
	};
};

#endif //SCENEFACTORY_H
