#ifndef ISCENE_H
#define ISCENE_H

#include "nau/attribute.h"
#include "nau/attributeValues.h"
#include "nau/event/iListener.h"
#include "nau/geometry/frustum.h"
#include "nau/scene/camera.h"
#include "nau/scene/light.h"

#include <string>
#include <vector>

namespace nau 
{
	namespace scene 
	{
		class SceneObject;

		class IScene : public AttributeValues, public IListener
		{
		public:
			FLOAT4_PROP(SCALE, 0);
			FLOAT4_PROP(ROTATE, 1);
			FLOAT4_PROP(TRANSLATE, 2);

			FLOAT3_PROP(BB_MIN, 0);
			FLOAT3_PROP(BB_MAX, 1);
			FLOAT3_PROP(BB_CENTER, 2);

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

		protected:

			IScene(void) : m_Compiled(false), m_Visible(true) {
				registerAndInitArrays(Attribs);
			};

			std::string m_Name;
			bool m_Compiled;
			//ITransform *m_Transform;
			mat4 m_Transform;
			bool m_Visible;
			std::set<std::string> m_MaterialNames;
			std::string m_Type;

			void updateTransform();

			static bool Init();
			static bool Inited;

		public:

			virtual void setPropf4(Float4Property prop, vec4& aVec);
			virtual void setPrope(EnumProperty prop, int v);
			vec3 &getPropf3(Float3Property prop);

			virtual void setName(std::string name) {
				m_Name = name; 
			};

			virtual std::string &getName() {
				return m_Name;
			};
		  
			virtual void add (SceneObject *aSceneObject) = 0;

			virtual void show (void) {m_Visible = true; }			
			virtual void hide (void) {m_Visible = false; }
			virtual bool isVisible (void) {return m_Visible; }

			virtual std::vector <SceneObject*>& findVisibleSceneObjects 
																(nau::geometry::Frustum &aFrustum, 
																Camera &aCamera, 
																bool conservative = false) = 0;
			virtual std::vector<SceneObject*>& getAllObjects (void) = 0;
			virtual SceneObject* getSceneObject (std::string name) = 0; 
			virtual SceneObject* getSceneObject (int index) = 0;

			virtual const std::set<std::string> &getMaterialNames() = 0;

			virtual void build (void) = 0;

			virtual void compile (void) = 0;
			bool isCompiled() { return m_Compiled;}

			virtual void unitize() = 0;

			virtual nau::math::mat4 &getTransform() = 0;
			virtual void setTransform(nau::math::mat4 &t) = 0;
			virtual void transform(nau::math::mat4 &t) = 0;

			virtual nau::geometry::IBoundingVolume& getBoundingVolume (void) = 0;

			virtual const std::string &getType (void);

			virtual ~IScene(void) {};

		};
	};
};

#endif
