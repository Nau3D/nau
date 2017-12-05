#ifndef GEOMETRYOBJECT_H
#define GEOMETRYOBJECT_H

/* These are very similiar to SceneObjects. It just simplifies the 
 material settings, since primitives have a single material with a fixed name */


#include "nau/scene/sceneObject.h"
#include "nau/scene/sceneObjectFactory.h"

namespace nau
{
	namespace scene
	{
		class GeometricObject : public nau::scene::SceneObject
		{
		public:
			friend class nau::scene::SceneObjectFactory;

			~GeometricObject(void);

			static unsigned int PrimitiveCounter;

			unsigned int m_PrimitiveID;
			void setRenderable(std::shared_ptr<IRenderable> &renderable);
			void setMaterial(const std::string &name);

			std::string getClassName(void);

		protected:
			GeometricObject();
			void eventReceived(const std::string & sender, const std::string & eventType, 
				const std::shared_ptr<IEventData>& evt);
		};
	};
};
#endif