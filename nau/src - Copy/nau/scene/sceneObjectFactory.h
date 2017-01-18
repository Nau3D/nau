#ifndef SCENEOBJECTFACTORY_H
#define SCENEOBJECTFACTORY_H

//#include "nau/scene/sceneObject.h"

#include <memory>
#include <string>


namespace nau
{

	namespace scene
	{
		class SceneObject;

		class SceneObjectFactory
		{

		public:
			//static nau::scene::SceneObject* Create (const std::string &type);
			static std::shared_ptr<SceneObject> Create(const std::string &type);
		private:
			//
			SceneObjectFactory(void) {};
			
		};
	};
};
#endif
