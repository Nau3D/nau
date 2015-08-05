#ifndef SCENEOBJECT_H
#define SCENEOBJECT_H

#include "nau/attribute.h"
#include "nau/attributeValues.h"
#include "nau/event/ilistener.h"
#include "nau/geometry/iBoundingVolume.h"
#include "nau/math/matrix.h"
#include "nau/render/iRenderable.h"
#include "nau/scene/sceneobjectfactory.h"

#include <string>

namespace nau
{
	namespace scene
	{
		class SceneObject : public AttributeValues, public nau::event_::IListener
		{
		public:
			friend class nau::scene::SceneObjectFactory;

			FLOAT4_PROP(SCALE, 0);
			FLOAT4_PROP(ROTATE, 1);
			FLOAT4_PROP(TRANSLATE, 2);

			ENUM_PROP(TRANSFORM_ORDER, 0);

			typedef enum {
				T_R_S,
				T_S_R,
				R_T_S,
				R_S_T,
				S_R_T,
				S_T_R
			} TransformationOrder;


			static AttribSet Attribs;

			virtual void setPropf4(Float4Property prop, vec4& aVec);

			static void ResetCounter();
			static unsigned int Counter;

			virtual std::string getType (void);

			virtual int getId ();
			virtual void setId (int id);

			virtual std::string& getName ();
			virtual void setName (const std::string &name);

			virtual void unitize(vec3 &center, vec3 &min, vec3 &max);

			virtual bool isStatic();
			virtual void setStaticCondition(bool aCondition);

			virtual nau::geometry::IBoundingVolume* getBoundingVolume();
			virtual void setBoundingVolume (nau::geometry::IBoundingVolume *b);

			virtual const nau::math::mat4& getTransform();
			virtual void transform(nau::math::mat4 &t);
			virtual void setTransform(nau::math::mat4 &t);
			virtual void burnTransform (void);
			virtual nau::math::mat4 *_getTransformPtr(void);
			virtual void updateGlobalTransform(nau::math::mat4 &m_Transform);
			
			virtual nau::render::IRenderable& getRenderable (void);
			virtual nau::render::IRenderable* _getRenderablePtr (void);
			virtual void setRenderable (nau::render::IRenderable *renderable);

			virtual void writeSpecificData (std::fstream &f);
			virtual void readSpecificData (std::fstream &f);

			void prepareTriangleIDs(bool ids);

			virtual ~SceneObject(void);


		protected:

			static bool Init();
			static bool Inited;

			SceneObject (void);
			int m_Id;
			std::string m_Name;
			bool m_StaticCondition;
			nau::render::IRenderable *m_Renderable;
			nau::geometry::IBoundingVolume *m_BoundingVolume;
			nau::math::mat4 m_Transform, m_GlobalTransform, m_ResultTransform;

			void calculateBoundingVolume (void);
			void updateTransform();
		};
	};
};

#endif // SCENEOBJECT_H
