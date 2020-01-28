#ifndef CAMERA_H
#define CAMERA_H

#include "nau/attribute.h"
#include "nau/attributeValues.h"
#include "nau/enums.h"
#include "nau/event/eventVec3.h"
#include "nau/math/matrix.h"
#include "nau/math/spherical.h"
#include "nau/math/vec4.h"
#include "nau/scene/sceneObject.h"
//#include "nau/render/renderManager.h"
#include "nau/render/viewport.h"

#include <memory>
#include <string>

using namespace nau::math;
using namespace nau::scene;

#ifdef _WINDLL
#ifdef nau_EXPORTS
#define nau_API __declspec(dllexport)   
#else  
#define nau_API __declspec(dllimport)   
#endif 
#else
#define nau_API
#endif


namespace nau
{
	namespace render
	{
		class RenderManager;
	}
	namespace scene
	{

		class Camera : public SceneObject
		{
			friend class nau::render::RenderManager;
		public:

			//std::shared_ptr<Camera> shared() {
			//	return std::shared_from_this();
			//}

			typedef enum {
				PERSPECTIVE,
				ORTHO
			} CameraType;

			FLOAT4_PROP(POSITION, 0);
			FLOAT4_PROP(VIEW_VEC ,1);
			FLOAT4_PROP(UP_VEC, 3);
			FLOAT4_PROP(NORMALIZED_UP_VEC, 4);
			FLOAT4_PROP(NORMALIZED_RIGHT_VEC, 5);
			FLOAT4_PROP(LOOK_AT_POINT, 6);
			FLOAT4_PROP(NORMALIZED_VIEW_VEC, 7);


			FLOAT_PROP(FOV, 0);
			FLOAT_PROP(NEARP, 1);
			FLOAT_PROP(FARP, 2);
			FLOAT_PROP(LEFT, 3);
			FLOAT_PROP(RIGHT, 4);
			FLOAT_PROP(TOP, 5);
			FLOAT_PROP(BOTTOM, 6);
			FLOAT_PROP(ELEVATION_ANGLE, 7);
			FLOAT_PROP(ZX_ANGLE, 8);

			MAT4_PROP(VIEW_MATRIX, 0);
			MAT4_PROP(PROJECTION_MATRIX, 1);
			MAT4_PROP(VIEW_INVERSE_MATRIX, 2);
			MAT4_PROP(PROJECTION_VIEW_MATRIX, 3);
			MAT4_PROP(TS05_PVM_MATRIX, 4);
			MAT4_PROP(PROJECTION_INVERSE_MATRIX, 5);

			ENUM_PROP(PROJECTION_TYPE, 0);

			STRING_PROP(VIEWPORT, 0);

			BOOL_PROP(DYNAMIC, 0);

			static nau_API AttribSet Attribs;
			static nau_API AttribSet &GetAttribs();

			virtual ~Camera (void);

			nau_API void setPropf4(Float4Property prop, vec4& aVec);
			nau_API void setPropf4(Float4Property prop, float r, float g, float b, float a);
			nau_API void setPropf(FloatProperty prop, float value);
			nau_API void setPropb(BoolProperty prop, bool value);
			nau_API void setPrope(EnumProperty prop, int value);
			nau_API void setProps(StringProperty prop, std::string &value);

			nau_API void setOrtho (float left, float right, float bottom, float top, float near, float far);
			nau_API void setPerspective (float fov, float near, float far);

			void eventReceived(const std::string &sender, const std::string &eventType, 
				const std::shared_ptr<IEventData> &evt);

			nau_API void setCamera (vec3 position, vec3 view, vec3 up);

			// Viewport
			nau_API std::shared_ptr<Viewport> &getViewport (void);
			nau_API void setViewport (std::shared_ptr<Viewport> aViewport);

			// Adjusts the frustum of the current Camera to include
			// the frustum of the target camera
			// usefull for shadow mapping for instance
			nau_API void adjustMatrix (std::shared_ptr<Camera> &targetCamera);
			// This version considers only a fraction of the target camera
			// the params near and far relate to the targets camera frustum
			nau_API void adjustMatrixPlus (float cNear, float cFar, std::shared_ptr<Camera> &targetCamera);

			// Bounding Volume 	
			nau_API virtual nau::geometry::IBoundingVolume* getBoundingVolume();

			// Renderable is the graphic representation of the camera
			// usefull for debug purposes
			nau_API std::shared_ptr<IRenderable> &getRenderable();

			// used for Physics
			//bool isDynamic();
			//void setDynamic(bool value);
			//void setPositionOffset (float value);

		private:
			Camera(const std::string &name);
			static std::shared_ptr<Camera> m_Temp;
		protected:

			static std::shared_ptr<Camera> Create(const std::string &name);
			static bool Init();
			static bool Inited;

			std::shared_ptr<Viewport> m_pViewport;

			vec3 result;
			// LookAt settings
			bool m_LookAt;

			// Camera Spherical Coordinates (angles are stored in radians)
			// Spherical(0,0) means Cartesian(0,0,1)
			void setVectorsFromSpherical();

			void updateProjection();

			// Matrices
			void buildViewMatrix (void);
			void buildInverses(void);
			void buildProjectionMatrix();
			void buildProjectionViewMatrix(void);
			void buildTS05PVMMatrix(void);

			// The eight corners of the frustum
			enum {
				TOP_LEFT_NEAR = 0,
				TOP_RIGHT_NEAR,
				BOTTOM_RIGHT_NEAR,
				BOTTOM_LEFT_NEAR,
				TOP_LEFT_FAR,
				TOP_RIGHT_FAR,
				BOTTOM_RIGHT_FAR,
				BOTTOM_LEFT_FAR,
			};		
		};
	};
};

#endif //CAMERA_H
