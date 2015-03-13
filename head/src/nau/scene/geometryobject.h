#ifndef GEOMETRYOBJECT_H
#define GEOMETRYOBJECT_H

/* These are very similiar to SceneObjects. It just simplifies the 
 material settings, since primitives have a single material with a fixed name */


#include "nau/scene/sceneobject.h"
#include "nau/scene/sceneobjectfactory.h"

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
			void setRenderable(nau::render::IRenderable *renderable);
			void setMaterial(const std::string &name);

			std::string getType (void);

		protected:
			GeometricObject();
		};
	};
};
#endif