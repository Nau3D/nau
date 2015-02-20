#ifndef PHYSICALOBJECT_H
#define PHYSICALOBJECT_H

#include <vector>

#include "nau/scene/sceneobject.h"

using namespace nau::scene;

namespace nau
{
	namespace world
	{
		class PhysicalObject
		{
		private:
			SceneObject *m_SceneObject;
		public:
			PhysicalObject (void);
		
			virtual ~PhysicalObject (void);
		};	
	};
};
#endif
