#ifndef QUAD_H
#define QUAD_H

#include "nau/scene/sceneObject.h"

namespace nau
{
	namespace geometry
	{
		class Quad : public nau::scene::SceneObject
		{
		public:
			Quad(void);
		public:
			~Quad(void);
			void eventReceived(const std::string & sender, const std::string & eventType, 
				const std::shared_ptr<IEventData>& evt);
		};
	};
};
#endif
