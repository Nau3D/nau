#ifndef IRENDERQUEUE_H
#define IRENDERQUEUE_H

#include "nau/material/materialId.h"
#include "nau/scene/sceneObject.h"

namespace nau
{
	namespace render
	{
		class IRenderQueue
		{
		public:
			virtual void clearQueue (void) = 0;
			virtual void addToQueue (nau::scene::SceneObject* aObject,
				std::map<std::string, nau::material::MaterialID> &materialMap) = 0;
			virtual void processQueue (void) = 0;

			virtual ~IRenderQueue(void) {};
		};
	};
};

#endif //IRENDERQUEUE_H
