#ifndef SCENEOBJECTFACTORY_H
#define SCENEOBJECTFACTORY_H

//#include <nau/scene/sceneobject.h>

#include <string>


namespace nau
{
	namespace scene
	{
		class SceneObject;

		class SceneObjectFactory
		{
		public:
			static nau::scene::SceneObject* create (std::string type);
		private:
			SceneObjectFactory(void) {};
			~SceneObjectFactory(void) {};
		};
	};
};
#endif
