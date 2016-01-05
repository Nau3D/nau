#ifndef SCENEFACTORY_H
#define SCENEFACTORY_H

#include "nau/scene/iScene.h"


namespace nau
{
	class render::RenderManager;

	namespace scene
	{
		
		class SceneFactory
		{
			friend class nau::render::RenderManager;
		public:
			/*static IScene * Create (std::string scene);*/
		private:
			static std::shared_ptr<IScene> Create(std::string name);
			SceneFactory(void) {};
		};
	};
};

#endif //SCENEFACTORY_H
